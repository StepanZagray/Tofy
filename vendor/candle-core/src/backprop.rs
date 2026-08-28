//! Methods for backpropagation of gradients.
use crate::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use crate::{Error, Result, Tensor, TensorId};
use std::collections::{hash_map::Entry, HashMap, HashSet};

// arg has been reduced to node via reduce_dims, expand it back to arg.
// This has to handle keepdims.
fn broadcast_back(arg: &Tensor, node: &Tensor, reduced_dims: &[usize]) -> Result<Tensor> {
    if arg.rank() == node.rank() {
        // keepdim = true
        node.broadcast_as(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        node.reshape(reduced_dims)?.broadcast_as(arg.shape())
    }
}

thread_local! {
    static CANDLE_GRAD_DO_NOT_DETACH: bool = {
        match std::env::var("CANDLE_GRAD_DO_NOT_DETACH") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    pub fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if node.dtype().is_int() {
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::IndexAdd(t1, t2, t3, _)
                    | Op::Scatter(t1, t2, t3, _)
                    | Op::ScatterAdd(t1, t2, t3, _)
                    | Op::CustomOp3(t1, t2, t3, _)
                    | Op::WhereCond(t1, t2, t3) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Conv1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::Conv2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::CustomOp2(lhs, rhs, _)
                    | Op::Binary(lhs, rhs, _)
                    | Op::Gather(lhs, rhs, _)
                    | Op::IndexSelect(lhs, rhs, _)
                    | Op::Matmul(lhs, rhs)
                    | Op::SliceScatter0(lhs, rhs, _) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Unary(_node, UnaryOp::Ceil)
                    | Op::Unary(_node, UnaryOp::Floor)
                    | Op::Unary(_node, UnaryOp::Round)
                    | Op::Unary(_node, UnaryOp::Sign) => nodes,
                    Op::Reshape(node)
                    | Op::UpsampleNearest1D { arg: node, .. }
                    | Op::UpsampleNearest2D { arg: node, .. }
                    | Op::UpsampleBilinear2D { arg: node, .. }
                    | Op::AvgPool2D { arg: node, .. }
                    | Op::MaxPool2D { arg: node, .. }
                    | Op::Copy(node)
                    | Op::Broadcast(node)
                    | Op::Cmp(node, _)
                    | Op::Reduce(node, ReduceOp::Min | ReduceOp::Sum | ReduceOp::Max, _)
                    | Op::ToDevice(node)
                    | Op::Transpose(node, _, _)
                    | Op::Permute(node, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Unary(node, _)
                    | Op::Elu(node, _)
                    | Op::Powf(node, _)
                    | Op::CustomOp1(node, _) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::ToDType(node) => {
                        if node.dtype().is_float() {
                            let (tg, nodes) = walk(node, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        } else {
                            nodes
                        }
                    }
                    Op::Reduce(_, ReduceOp::ArgMin | ReduceOp::ArgMax, _) => nodes,
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore> {
        let sorted_nodes = self.sorted_nodes();
        // Nodes that participate in the gradient graph (Vars + float ancestors of the
        // loss). Candle's stock Conv{1,2}D backward always materializes `grad_arg` even
        // when `arg` is a non-tracking leaf (e.g. one-hot pixels). Skipping that dead
        // work leaves weight gradients unchanged.
        let needs_grad: HashSet<TensorId> = sorted_nodes.iter().map(|t| t.id()).collect();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?.contiguous()?);
        for node in sorted_nodes.iter() {
            if node.is_variable() {
                continue;
            }
            let grad = grads
                .remove(node)
                .expect("candle internal error - grad not populated");
            // https://github.com/huggingface/candle/issues/1241
            // Ideally, we would make these operations in place where possible to ensure that we
            // do not have to allocate too often. Here we just call `.detach` to avoid computing
            // the backprop graph of the backprop itself. This would be an issue for second order
            // derivatives but these are out of scope at the moment.
            let do_not_detach = CANDLE_GRAD_DO_NOT_DETACH.with(|b| *b);
            let grad = if do_not_detach { grad } else { grad.detach() };
            if let Some(op) = node.op() {
                match op {
                    Op::Binary(lhs, rhs, BinaryOp::Add) => {
                        if needs_grad.contains(&lhs.id()) {
                            grads.accumulate(lhs, grad.clone())?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            grads.accumulate(rhs, grad.clone())?;
                        }
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Sub) => {
                        if needs_grad.contains(&lhs.id()) {
                            grads.accumulate(lhs, grad.clone())?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            grads.accumulate_sub(rhs, grad.clone())?;
                        }
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                        if needs_grad.contains(&lhs.id()) {
                            grads.accumulate(lhs, grad.mul(rhs)?)?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            grads.accumulate(rhs, grad.mul(lhs)?)?;
                        }
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Div) => {
                        if needs_grad.contains(&lhs.id()) {
                            grads.accumulate(lhs, grad.div(rhs)?)?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            grads.accumulate_sub(rhs, grad.mul(lhs)?.div(&rhs.sqr()?)?)?;
                        }
                    }
                    Op::Binary(lhs, rhs, BinaryOp::Minimum)
                    | Op::Binary(lhs, rhs, BinaryOp::Maximum) => {
                        let mask_lhs = node.eq(lhs)?.to_dtype(grad.dtype())?;
                        let mask_rhs = node.eq(rhs)?.to_dtype(grad.dtype())?;

                        // If both masks are 1 one the same point, we want to scale the
                        // gradient by 0.5 rather than 1.
                        if needs_grad.contains(&lhs.id()) {
                            let lhs_grad = mask_lhs.mul(&grad)?.div(&(&mask_rhs + 1.)?)?;
                            grads.accumulate(lhs, lhs_grad)?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            let rhs_grad = mask_rhs.mul(&grad)?.div(&(&mask_lhs + 1.)?)?;
                            grads.accumulate(rhs, rhs_grad)?;
                        }
                    }
                    Op::WhereCond(pred, t, f) => {
                        let zeros = grad.zeros_like()?;
                        if needs_grad.contains(&t.id()) {
                            grads.accumulate(t, pred.where_cond(&grad, &zeros)?)?;
                        }
                        if needs_grad.contains(&f.id()) {
                            grads.accumulate(f, pred.where_cond(&zeros, &grad)?)?;
                        }
                    }
                    Op::Conv1D {
                        arg,
                        kernel,
                        padding,
                        stride,
                        dilation,
                    } => {
                        // The output height for conv_transpose1d is:
                        // (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1
                        if needs_grad.contains(&arg.id()) {
                            let grad_l_in = grad.dim(2)?;
                            let k_size = kernel.dim(2)?;
                            let out_size = (grad_l_in - 1) * stride + dilation * (k_size - 1) + 1
                                - 2 * padding;
                            let out_padding = arg.dim(2)? - out_size;
                            let grad_arg = grad.conv_transpose1d(
                                kernel,
                                *padding,
                                out_padding,
                                *stride,
                                *dilation,
                                /* groups */ 1,
                            )?;
                            grads.accumulate(arg, grad_arg)?;
                        }

                        if needs_grad.contains(&kernel.id()) {
                            let grad_kernel = arg
                                .transpose(0, 1)?
                                .conv1d(&grad.transpose(0, 1)?, *padding, *dilation, *stride, 1)?
                                .transpose(0, 1)?;
                            let (_, _, k0) = kernel.dims3()?;
                            let (_, _, g_k0) = grad_kernel.dims3()?;
                            let grad_kernel = if g_k0 != k0 {
                                grad_kernel.narrow(2, 0, k0)?
                            } else {
                                grad_kernel
                            };
                            grads.accumulate(kernel, grad_kernel)?;
                        }
                    }
                    Op::Conv2D {
                        arg,
                        kernel,
                        padding,
                        stride,
                        dilation,
                    } => {
                        let use_cudnn_bwd = {
                            #[cfg(all(feature = "cuda", feature = "cudnn"))]
                            {
                                arg.device().is_cuda()
                                    && kernel.layout().is_contiguous()
                                    && arg.layout().is_contiguous()
                            }
                            #[cfg(not(all(feature = "cuda", feature = "cudnn")))]
                            {
                                false
                            }
                        };
                        let need_arg = needs_grad.contains(&arg.id());
                        let need_kernel = needs_grad.contains(&kernel.id());

                        if use_cudnn_bwd {
                            #[cfg(all(feature = "cuda", feature = "cudnn"))]
                            {
                                if need_arg {
                                    let grad_arg = arg.conv2d_bwd_data(
                                        &grad, kernel, *padding, *stride, *dilation,
                                    )?;
                                    grads.accumulate(arg, grad_arg)?;
                                }
                                if need_kernel {
                                    let grad_kernel = arg.conv2d_bwd_filter(
                                        &grad, kernel, *padding, *stride, *dilation,
                                    )?;
                                    let (_, _, k0, k1) = kernel.dims4()?;
                                    let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
                                    let grad_kernel = if g_k0 != k0 || g_k1 != k1 {
                                        grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
                                    } else {
                                        grad_kernel
                                    };
                                    grads.accumulate(kernel, grad_kernel)?;
                                }
                            }
                        } else {
                            // The output height for conv_transpose2d is:
                            // (i_h - 1) * stride - 2 * padding + dilation * (k_h - 1) + out_padding + 1
                            if need_arg {
                                let grad_h = grad.dim(2)?;
                                let k_h = kernel.dim(2)?;
                                let out_size =
                                    (grad_h - 1) * stride + dilation * (k_h - 1) + 1 - 2 * padding;
                                let out_padding = arg.dim(2)? - out_size;
                                let grad_arg = grad.conv_transpose2d(
                                    kernel,
                                    *padding,
                                    out_padding,
                                    *stride,
                                    *dilation,
                                )?;
                                grads.accumulate(arg, grad_arg)?;
                            }
                            if need_kernel {
                                let grad_kernel = arg
                                    .transpose(0, 1)?
                                    .conv2d(
                                        &grad.transpose(0, 1)?,
                                        *padding,
                                        *dilation,
                                        *stride,
                                        1,
                                    )?
                                    .transpose(0, 1)?;
                                let (_, _, k0, k1) = kernel.dims4()?;
                                let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
                                let grad_kernel = if g_k0 != k0 || g_k1 != k1 {
                                    grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
                                } else {
                                    grad_kernel
                                };
                                grads.accumulate(kernel, grad_kernel)?;
                            }
                        }
                    }
                    Op::ConvTranspose1D { .. } => Err(Error::BackwardNotSupported {
                        op: "conv-transpose1d",
                    })?,
                    Op::ConvTranspose2D {
                        arg,
                        kernel,
                        padding,
                        stride,
                        dilation,
                        output_padding: _output_padding,
                    } => {
                        if needs_grad.contains(&arg.id()) {
                            let grad_arg = grad.conv2d(kernel, *padding, *stride, *dilation, 1)?;
                            grads.accumulate(arg, grad_arg)?;
                        }
                        if needs_grad.contains(&kernel.id()) {
                            let grad_kernel = grad
                                .transpose(0, 1)?
                                .conv2d(&arg.transpose(0, 1)?, *padding, *dilation, *stride, 1)?
                                .transpose(0, 1)?;
                            let (_, _, k0, k1) = kernel.dims4()?;
                            let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
                            let grad_kernel = if g_k0 != k0 || g_k1 != k1 {
                                grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
                            } else {
                                grad_kernel
                            };
                            grads.accumulate(kernel, grad_kernel)?;
                        }
                    }
                    Op::AvgPool2D {
                        arg,
                        kernel_size,
                        stride,
                    } => {
                        if kernel_size != stride {
                            crate::bail!("backward not supported for avgpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        let (_n, _c, h, w) = arg.dims4()?;
                        let grad_arg = grad.upsample_nearest2d(h, w)?;
                        let grad_arg =
                            (grad_arg * (1f64 / (kernel_size.0 * kernel_size.1) as f64))?;
                        grads.accumulate(arg, grad_arg)?;
                    }
                    Op::MaxPool2D {
                        arg,
                        kernel_size,
                        stride,
                    } => {
                        if kernel_size != stride {
                            crate::bail!("backward not supported for maxpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        let (_n, _c, h, w) = arg.dims4()?;
                        // For computing the max-pool gradient, we compute a mask where a 1 means
                        // that the element is the maximum, then we apply this mask to the
                        // upsampled gradient (taking into account that multiple max may exist so
                        // we scale the gradient for this case).
                        let node_upsampled = node.upsample_nearest2d(h, w)?;
                        let mask = arg.eq(&node_upsampled)?.to_dtype(arg.dtype())?;
                        let avg = mask.avg_pool2d_with_stride(*kernel_size, *stride)?;
                        let grad_arg = ((grad * avg)?.upsample_nearest2d(h, w)? * mask)?;
                        grads.accumulate(arg, grad_arg)?;
                    }
                    Op::UpsampleNearest1D { arg, target_size } => {
                        let (_n, c, size) = arg.dims3()?;
                        if target_size % size != 0 {
                            crate::bail!("backward not supported for non integer upscaling factors")
                        }
                        let scale = target_size / size;

                        let kernel = Tensor::ones((c, 1, scale), arg.dtype(), arg.device())?;
                        let conv_sum = grad.conv1d(&kernel, 0, scale, 1, c)?;
                        grads.accumulate(arg, conv_sum)?;
                    }
                    Op::UpsampleNearest2D {
                        arg,
                        target_h,
                        target_w,
                    } => {
                        let (_n, c, h, w) = arg.dims4()?;
                        if target_h % h != 0 || target_w % w != 0 {
                            crate::bail!("backward not supported for non integer upscaling factors")
                        }
                        let scale_h = target_h / h;
                        let scale_w = target_w / w;

                        if scale_h != scale_w {
                            crate::bail!("backward not supported for non uniform upscaling factors")
                        };
                        let kernel =
                            Tensor::ones((c, 1, scale_h, scale_w), arg.dtype(), arg.device())?;
                        let conv_sum = grad.conv2d(&kernel, 0, scale_h, 1, c)?;
                        grads.accumulate(arg, conv_sum)?;
                    }
                    Op::UpsampleBilinear2D { .. } => {
                        crate::bail!("backward not supported for upsample_bilinear2d")
                    }
                    Op::SliceScatter0(lhs, rhs, start_rhs) => {
                        if needs_grad.contains(&rhs.id()) {
                            let rhs_grad = grad.narrow(0, *start_rhs, rhs.dim(0)?)?;
                            grads.accumulate(rhs, rhs_grad)?;
                        }
                        if needs_grad.contains(&lhs.id()) {
                            let lhs_grad = grad.slice_scatter0(&rhs.zeros_like()?, *start_rhs)?;
                            grads.accumulate(lhs, lhs_grad)?;
                        }
                    }
                    Op::Gather(arg, indexes, dim) => {
                        let arg_grad = arg.zeros_like()?.scatter_add(indexes, &grad, *dim)?;
                        grads.accumulate(arg, arg_grad)?;
                    }
                    Op::Scatter(init, indexes, src, dim) => {
                        if needs_grad.contains(&init.id()) {
                            grads.accumulate(init, grad.clone())?;
                        }
                        if needs_grad.contains(&src.id()) {
                            grads.accumulate(src, grad.gather(indexes, *dim)?)?;
                        }
                    }
                    Op::ScatterAdd(init, indexes, src, dim) => {
                        if needs_grad.contains(&init.id()) {
                            let mask = init.ones_like()?;
                            let mask = mask.scatter(indexes, &mask.zeros_like()?, *dim)?;
                            grads.accumulate(init, grad.mul(&mask)?)?;
                        }
                        if needs_grad.contains(&src.id()) {
                            grads.accumulate(src, grad.gather(indexes, *dim)?)?;
                        }
                    }
                    Op::IndexAdd(init, indexes, src, dim) => {
                        if needs_grad.contains(&init.id()) {
                            grads.accumulate(init, grad.clone())?;
                        }
                        if needs_grad.contains(&src.id()) {
                            grads.accumulate(src, grad.index_select(indexes, *dim)?)?;
                        }
                    }
                    Op::IndexSelect(arg, indexes, dim) => {
                        let arg_grad = arg.zeros_like()?.index_add(indexes, &grad, *dim)?;
                        grads.accumulate(arg, arg_grad)?;
                    }
                    Op::Matmul(lhs, rhs) => {
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.
                        if needs_grad.contains(&lhs.id()) {
                            grads.accumulate(lhs, grad.matmul(&rhs.t()?)?)?;
                        }
                        if needs_grad.contains(&rhs.id()) {
                            grads.accumulate(rhs, lhs.t()?.matmul(&grad)?)?;
                        }
                    }
                    Op::Cat(args, dim) => {
                        let mut start_idx = 0;
                        for arg in args {
                            let len = arg.dims()[*dim];
                            if needs_grad.contains(&arg.id()) {
                                let arg_grad = grad.narrow(*dim, start_idx, len)?;
                                grads.accumulate(arg, arg_grad)?;
                            }
                            start_idx += len;
                        }
                    }
                    Op::Broadcast(arg) => {
                        if !needs_grad.contains(&arg.id()) {
                            continue;
                        }
                        let arg_dims = arg.dims();
                        let node_dims = node.dims();
                        // The number of dims that have been inserted on the left.
                        let left_dims = node_dims.len() - arg_dims.len();
                        let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                        for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                            .iter()
                            .zip(arg_dims.iter())
                            .enumerate()
                        {
                            if node_dim != arg_dim {
                                sum_dims.push(dim + left_dims)
                            }
                        }

                        let mut arg_grad = grad.sum_keepdim(sum_dims.as_slice())?;
                        for _i in 0..left_dims {
                            arg_grad = arg_grad.squeeze(0)?
                        }
                        grads.accumulate(arg, arg_grad.broadcast_as(arg.dims())?)?;
                    }
                    Op::Reduce(arg, ReduceOp::Sum, reduced_dims) => {
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        grads.accumulate(arg, grad)?;
                    }
                    Op::Reduce(arg, ReduceOp::Max, reduced_dims) => {
                        let node = broadcast_back(arg, node, reduced_dims)?;
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                        grads.accumulate(arg, grad.broadcast_as(arg.dims())?)?;
                    }
                    Op::Reduce(arg, ReduceOp::Min, reduced_dims) => {
                        let node = broadcast_back(arg, node, reduced_dims)?;
                        let grad = broadcast_back(arg, &grad, reduced_dims)?;
                        let grad = node.eq(arg)?.to_dtype(grad.dtype())?.mul(&grad)?;
                        grads.accumulate(arg, grad.broadcast_as(arg.dims())?)?;
                    }
                    Op::ToDType(arg) => grads.accumulate(arg, grad.to_dtype(arg.dtype())?)?,
                    Op::Copy(arg) => grads.accumulate(arg, grad)?,
                    Op::Affine { arg, mul, .. } => {
                        let arg_grad = grad.affine(*mul, 0.)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Unary(arg, UnaryOp::Log) => grads.accumulate(arg, (grad / arg)?)?,
                    Op::Unary(arg, UnaryOp::Sin) => grads.accumulate(arg, (&grad * arg.cos())?)?,
                    Op::Unary(arg, UnaryOp::Cos) => {
                        grads.accumulate_sub(arg, (&grad * arg.sin())?)?
                    }
                    Op::Unary(arg, UnaryOp::Tanh) => {
                        let minus_dtanh = (node.sqr()? - 1.)?;
                        grads.accumulate_sub(arg, (&grad * &minus_dtanh)?)?
                    }
                    Op::Unary(arg, UnaryOp::Abs) => {
                        let ones = arg.ones_like()?;
                        let abs_grad = arg.ge(&arg.zeros_like()?)?.where_cond(&ones, &ones.neg()?);
                        grads.accumulate(arg, (&grad * abs_grad)?)?
                    }
                    Op::Unary(arg, UnaryOp::Exp) => grads.accumulate(arg, (&grad * *node)?)?,
                    Op::Unary(arg, UnaryOp::Neg) => grads.accumulate_sub(arg, grad)?,
                    Op::Unary(arg, UnaryOp::Recip) => {
                        let grad = (grad / arg.sqr()?)?;
                        grads.accumulate_sub(arg, grad)?
                    }
                    &Op::Narrow(ref arg, dim, start_idx, len) => {
                        let arg_dims = arg.dims();
                        let left_pad = if start_idx == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = start_idx;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let right_pad = arg_dims[dim] - start_idx - len;
                        let right_pad = if right_pad == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = right_pad;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let arg_grad = match (left_pad, right_pad) {
                            (None, None) => grad,
                            (Some(l), None) => Tensor::cat(&[&l, &grad], dim)?,
                            (None, Some(r)) => Tensor::cat(&[&grad, &r], dim)?,
                            (Some(l), Some(r)) => Tensor::cat(&[&l, &grad, &r], dim)?,
                        };
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Unary(_, UnaryOp::Floor)
                    | Op::Unary(_, UnaryOp::Round)
                    | Op::Reduce(_, ReduceOp::ArgMin, _)
                    | Op::Reduce(_, ReduceOp::ArgMax, _)
                    | Op::Unary(_, UnaryOp::Sign)
                    | Op::Cmp(_, _) => {}
                    Op::Reshape(arg) => {
                        let arg_grad = grad.reshape(arg.dims())?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Unary(_, UnaryOp::Ceil) => Err(Error::BackwardNotSupported { op: "ceil" })?,
                    Op::Unary(arg, UnaryOp::Gelu) => {
                        let cube = arg.powf(3.)?;
                        let tanh = (0.0356774 * &cube + (0.797885 * arg)?)?.tanh()?;
                        let gelu_grad = (((0.5 * &tanh)?
                            + (0.0535161 * cube + (0.398942 * arg)?)? * (1. - tanh.powf(2.)?))?
                            + 0.5)?;
                        grads.accumulate(arg, (&grad * gelu_grad)?)?
                    }
                    Op::Unary(arg, UnaryOp::Erf) => {
                        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                        let erf_grad =
                            (2. / std::f64::consts::PI.sqrt()) * (arg.sqr()?.neg()?).exp()?;
                        grads.accumulate(arg, (&grad * erf_grad)?)?
                    }
                    Op::Unary(arg, UnaryOp::GeluErf) => {
                        // d/dx gelu_erf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                        let neg_half_square = (arg.sqr()?.neg()? / 2.)?;
                        let scaled_exp_arg = (0.398942 * neg_half_square.exp()? * arg)?;
                        let arg_scaled_sqrt = (arg / 2f64.sqrt())?;
                        let erf_scaled_sqrt = (0.5 * arg_scaled_sqrt.erf()?)?;
                        let gelu_erf_grad = (0.5 + scaled_exp_arg + erf_scaled_sqrt)?;
                        grads.accumulate(arg, (&grad * gelu_erf_grad)?)?;
                    }
                    Op::Unary(arg, UnaryOp::Relu) => {
                        let relu_grad = arg.ge(&arg.zeros_like()?)?.to_dtype(arg.dtype())?;
                        grads.accumulate(arg, (&grad * relu_grad)?)?
                    }
                    Op::Unary(arg, UnaryOp::Silu) => {
                        // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                        let sigmoid_arg = (arg.neg()?.exp()? + 1.)?.recip()?;
                        let silu_grad = &sigmoid_arg * (1. - *node) + *node;
                        grads.accumulate(arg, (&grad * silu_grad)?)?
                    }
                    Op::Elu(arg, alpha) => {
                        // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                        let zeros = arg.zeros_like()?;
                        let positive_mask = arg.gt(&zeros)?.to_dtype(arg.dtype())?;
                        let negative_mask = arg.le(&zeros)?.to_dtype(arg.dtype())?;
                        // node == alpha * (e^x - 1) for x <= 0, reuse it
                        let negative_exp_mask = (negative_mask * (*node + *alpha))?;
                        let combined_mask = (positive_mask + negative_exp_mask)?;
                        grads.accumulate(arg, (grad * combined_mask)?)?
                    }
                    Op::Powf(arg, e) => {
                        let arg_grad = (&(grad * arg.powf(e - 1.)?)? * *e)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::CustomOp1(arg, c) => {
                        if let Some(arg_grad) = c.bwd(arg, node, &grad)? {
                            grads.accumulate(arg, arg_grad)?
                        }
                    }
                    Op::CustomOp2(arg1, arg2, c) => {
                        let (arg_grad1, arg_grad2) = c.bwd(arg1, arg2, node, &grad)?;
                        if needs_grad.contains(&arg1.id()) {
                            if let Some(arg_grad1) = arg_grad1 {
                                grads.accumulate(arg1, arg_grad1)?
                            }
                        }
                        if needs_grad.contains(&arg2.id()) {
                            if let Some(arg_grad2) = arg_grad2 {
                                grads.accumulate(arg2, arg_grad2)?
                            }
                        }
                    }
                    Op::CustomOp3(arg1, arg2, arg3, c) => {
                        let (arg_grad1, arg_grad2, arg_grad3) =
                            c.bwd(arg1, arg2, arg3, node, &grad)?;
                        if needs_grad.contains(&arg1.id()) {
                            if let Some(arg_grad1) = arg_grad1 {
                                grads.accumulate(arg1, arg_grad1)?
                            }
                        }
                        if needs_grad.contains(&arg2.id()) {
                            if let Some(arg_grad2) = arg_grad2 {
                                grads.accumulate(arg2, arg_grad2)?
                            }
                        }
                        if needs_grad.contains(&arg3.id()) {
                            if let Some(arg_grad3) = arg_grad3 {
                                grads.accumulate(arg3, arg_grad3)?
                            }
                        }
                    }
                    Op::Unary(arg, UnaryOp::Sqr) => {
                        let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Unary(arg, UnaryOp::Sqrt) => {
                        let arg_grad = grad.div(node)?.affine(0.5, 0.)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::ToDevice(arg) => {
                        let arg_grad = grad.to_device(arg.device())?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Transpose(arg, dim1, dim2) => {
                        let arg_grad = grad.transpose(*dim1, *dim2)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                    Op::Permute(arg, dims) => {
                        let mut inv_dims = vec![0; dims.len()];
                        for (i, &dim_idx) in dims.iter().enumerate() {
                            inv_dims[dim_idx] = i
                        }
                        let arg_grad = grad.permute(inv_dims)?;
                        grads.accumulate(arg, arg_grad)?
                    }
                };
            }
        }
        Ok(grads)
    }
}

/// A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
#[derive(Default, Debug)]
pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    /// Create a new gradient store
    fn new() -> Self {
        Self::default()
    }

    /// Get the gradient tensor corresponding to the given tensor id
    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    /// Get the gradient tensor associated with the given tensor
    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    /// Remove the gradient tensor associated with the given tensor, returning it if it exists
    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    /// Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

    /// Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
    pub fn insert_id(&mut self, id: TensorId, grad: Tensor) -> Option<Tensor> {
        self.0.insert(id, grad)
    }

    /// Accumulate one contribution. The stored gradient is always contiguous,
    /// matching the layout produced by adding into a same-shaped zero tensor.
    fn accumulate(&mut self, tensor: &Tensor, grad: Tensor) -> Result<()> {
        match self.0.entry(tensor.id()) {
            Entry::Occupied(mut entry) => {
                let grad = entry.get().add(&grad)?;
                *entry.get_mut() = grad;
            }
            Entry::Vacant(entry) => {
                entry.insert(grad.contiguous()?);
            }
        }
        Ok(())
    }

    /// Accumulate the negation of one contribution without materializing a
    /// same-shaped zero tensor for the first contribution.
    fn accumulate_sub(&mut self, tensor: &Tensor, grad: Tensor) -> Result<()> {
        match self.0.entry(tensor.id()) {
            Entry::Occupied(mut entry) => {
                let grad = entry.get().sub(&grad)?;
                *entry.get_mut() = grad;
            }
            Entry::Vacant(entry) => {
                entry.insert(grad.neg()?.contiguous()?);
            }
        }
        Ok(())
    }

    /// Extend this gradient store with the contents of another.
    /// If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
    /// Otherwise simply inserts.
    pub fn extend(&mut self, other: Self) -> Result<()> {
        for (id, grad) in other.0 {
            match self.0.entry(id) {
                Entry::Occupied(mut entry) => {
                    let new_grad = entry.get().add(&grad)?;

                    let do_not_detach = CANDLE_GRAD_DO_NOT_DETACH.with(|b| *b);
                    let new_grad = if do_not_detach {
                        new_grad
                    } else {
                        new_grad.detach()
                    };

                    *entry.get_mut() = new_grad;
                }
                Entry::Vacant(entry) => {
                    entry.insert(grad);
                }
            }
        }
        Ok(())
    }

    /// Get the tensor ids of the stored gradient tensors
    pub fn get_ids(&self) -> impl Iterator<Item = &TensorId> {
        self.0.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device, Var};

    fn multipath_loss(x: &Tensor, kernel: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let conv = x.conv2d(kernel, 0, 1, 1, 1)?;
        let flat = conv.permute((0, 2, 3, 1))?.reshape((4, 1))?;
        let projected = flat.matmul(weight)?;
        let bias = Tensor::new(&[0.25f32, -0.5], x.device())?;
        let scale = Tensor::new(&[1.5f32, -0.75], x.device())?;
        let first = projected
            .broadcast_add(&bias)?
            .broadcast_mul(&scale)?
            .narrow(0, 0, 2)?;
        let second = projected.narrow(0, 2, 2)?;
        let woven = Tensor::cat(&[&first, &second], 0)?
            .reshape((2, 2, 2))?
            .permute((1, 0, 2))?;
        woven
            .sum_all()?
            .add(&flat.mul(&flat)?.sum_all()?.affine(0.125, 0.0)?)
    }

    fn eval_multipath(x: &[f32], kernel: &[f32], weight: &[f32]) -> Result<f32> {
        let device = Device::Cpu;
        multipath_loss(
            &Tensor::new(x, &device)?.reshape((1, 1, 3, 3))?,
            &Tensor::new(kernel, &device)?.reshape((1, 1, 2, 2))?,
            &Tensor::new(weight, &device)?.reshape((1, 2))?,
        )?
        .to_scalar::<f32>()
    }

    fn assert_finite_difference(
        base: &[f32],
        analytic: &[f32],
        mut eval: impl FnMut(&[f32]) -> Result<f32>,
    ) -> Result<()> {
        let eps = 1e-3f32;
        for index in 0..base.len() {
            let mut plus = base.to_vec();
            let mut minus = base.to_vec();
            plus[index] += eps;
            minus[index] -= eps;
            let numerical = (eval(&plus)? - eval(&minus)?) / (2.0 * eps);
            assert!(
                (analytic[index] - numerical).abs() < 7e-3,
                "gradient {index}: analytic={} numerical={numerical}",
                analytic[index]
            );
        }
        Ok(())
    }

    #[test]
    fn first_gradient_and_multipath_accumulation_match_finite_differences() -> Result<()> {
        let device = Device::Cpu;
        let x_values = vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.5, -0.8, 0.9, 0.6];
        let kernel_values = vec![0.3, -0.2, 0.4, 0.6];
        let weight_values = vec![0.75, -1.25];
        let x =
            Var::from_tensor(&Tensor::new(x_values.as_slice(), &device)?.reshape((1, 1, 3, 3))?)?;
        let kernel = Var::from_tensor(
            &Tensor::new(kernel_values.as_slice(), &device)?.reshape((1, 1, 2, 2))?,
        )?;
        let weight =
            Var::from_tensor(&Tensor::new(weight_values.as_slice(), &device)?.reshape((1, 2))?)?;
        let grads = multipath_loss(&x, &kernel, &weight)?.backward()?;
        let x_grad = grads
            .get(&x)
            .expect("x gradient")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let kernel_grad = grads
            .get(&kernel)
            .expect("kernel gradient")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let weight_grad = grads
            .get(&weight)
            .expect("weight gradient")
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_finite_difference(&x_values, &x_grad, |values| {
            eval_multipath(values, &kernel_values, &weight_values)
        })?;
        assert_finite_difference(&kernel_values, &kernel_grad, |values| {
            eval_multipath(&x_values, values, &weight_values)
        })?;
        assert_finite_difference(&weight_values, &weight_grad, |values| {
            eval_multipath(&x_values, &kernel_values, values)
        })?;

        let non_contiguous =
            Var::from_tensor(&Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?)?;
        let permuted_loss = non_contiguous.transpose(0, 1)?.sum_all()?;
        let permuted_grads = permuted_loss.backward()?;
        assert!(
            permuted_grads
                .get(&non_contiguous)
                .expect("permuted gradient")
                .layout()
                .is_contiguous(),
            "the first stored contribution must retain the old contiguous layout"
        );

        let subtracted = Var::new(&[2f32, -3.], &device)?;
        let zero = Tensor::zeros((2,), DType::F32, &device)?;
        let subtracted_grads = zero.sub(&subtracted)?.sum_all()?.backward()?;
        assert_eq!(
            subtracted_grads
                .get(&subtracted)
                .expect("subtracted gradient")
                .to_vec1::<f32>()?,
            vec![-1.0, -1.0]
        );
        assert!(subtracted_grads.get(&zero).is_none());
        Ok(())
    }

    #[test]
    fn constant_operands_are_not_materialized_and_var_paths_are_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::new(&[[1f32, 2.], [3., 4.]], &device)?;
        let constant = Tensor::new(&[[0.5f32, -1.], [2., 0.25]], &device)?;
        let rhs = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
        let cat_constant = Tensor::new(&[[8f32, 9.], [10., 11.]], &device)?;
        let false_value = Tensor::zeros((2, 2), DType::F32, &device)?;
        let predicate = Tensor::new(&[[1u8, 0], [0, 1]], &device)?;
        let row = Tensor::new(&[3f32, -2.], &device)?;
        let broadcast = row.broadcast_as((2, 2))?;

        let binary = var.add(&constant)?.mul(&constant)?.sum_all()?;
        let matmul = var.matmul(&rhs)?.sum_all()?;
        let cat = Tensor::cat(&[var.as_tensor(), &cat_constant], 0)?.sum_all()?;
        let selected = predicate.where_cond(&var, &false_value)?.sum_all()?;
        let broadcasted = var.add(&broadcast)?.sum_all()?;
        let loss = binary
            .add(&matmul)?
            .add(&cat)?
            .add(&selected)?
            .add(&broadcasted)?;
        let grads = loss.backward()?;

        assert_eq!(
            grads.get(&var).expect("var gradient").to_vec2::<f32>()?,
            vec![vec![6.5, 8.0], vec![7.0, 10.25]]
        );
        for tensor in [&constant, &rhs, &cat_constant, &false_value, &broadcast] {
            assert!(
                grads.get(tensor).is_none(),
                "constant operand unexpectedly received a gradient"
            );
        }
        Ok(())
    }

    #[test]
    fn conv2d_skips_each_constant_operand_without_changing_the_other_gradient() -> Result<()> {
        let device = Device::Cpu;
        let input_values = Tensor::new(&[[[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]]]], &device)?;
        let kernel_values = Tensor::ones((1, 1, 2, 2), DType::F32, &device)?;

        let input = Var::from_tensor(&input_values)?;
        let input_grads = input
            .conv2d(&kernel_values, 0, 1, 1, 1)?
            .sum_all()?
            .backward()?;
        assert_eq!(
            input_grads
                .get(&input)
                .expect("input gradient")
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1., 2., 1., 2., 4., 2., 1., 2., 1.]
        );
        assert!(input_grads.get(&kernel_values).is_none());

        let kernel = Var::from_tensor(&kernel_values)?;
        let kernel_grads = input_values
            .conv2d(&kernel, 0, 1, 1, 1)?
            .sum_all()?
            .backward()?;
        assert_eq!(
            kernel_grads
                .get(&kernel)
                .expect("kernel gradient")
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![12., 16., 24., 28.]
        );
        assert!(kernel_grads.get(&input_values).is_none());
        Ok(())
    }
}
