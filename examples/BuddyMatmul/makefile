#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
LIB_OMP := ${LLVM_BUILD_DIR}/lib/libomp.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

linalg-batchmatmul-f32-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batchmatmul-vectorization-lower:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-o ./log.mlir

linalg-batchmatmul-f32-vectorization-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batchmatmul-f32-omp-lower:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts  \
		-o log.mlir

linalg-batchmatmul-f32-omp-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${LIB_OMP}

linalg-matmul-transpose-b-f32-run:
	@${BUDDY_OPT} ./linalg-transpose-matmul-b-f32.mlir\
		-matmul-transpose-b-vectorization \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2.mlir \
			-pass-pipeline "builtin.module(transpose-fusion-vectorization, func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
    ${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-tensor-to-linalg \
		-func-bufferize \
		-arith-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-vec-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2-vec.mlir\
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm  \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
