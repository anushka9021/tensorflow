/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <iostream>
#include <vector>
#include <string>
#include <omp.h>  // For parallel processing (optional)
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

constexpr int kCompressedElementVersion = 0;
constexpr size_t kMaxSnappySize = kuint32max; // 4GB Snappy limit

class Iov {
 public:
  explicit Iov(size_t size) : iov_(size), idx_(0), num_bytes_(0) {}

  void Add(void* base, size_t len) {
    iov_[idx_].iov_base = base;
    iov_[idx_].iov_len = len;
    num_bytes_ += len;
    ++idx_;
  }

  iovec* Data() { return iov_.data(); }

  size_t NumBytes() const { return num_bytes_; }

  size_t NumPieces() const { return iov_.size(); }

 private:
  std::vector<struct iovec> iov_;
  size_t idx_;
  size_t num_bytes_;
};

Status CompressElement(const std::vector<Tensor>& element,
                       CompressedElement* out) {
  size_t num_string_tensors = 0;
  size_t num_string_tensor_strings = 0;
  std::vector<TensorProto> nonmemcpyable_components;
  size_t total_nonmemcpyable_size = 0;

  // First pass: collect non-memcpyable tensor info
  for (const auto& component : element) {
    if (component.dtype() == DT_STRING) {
      ++num_string_tensors;
      num_string_tensor_strings += component.NumElements();
    } else if (!DataTypeCanUseMemcpy(component.dtype())) {
      nonmemcpyable_components.emplace_back();
      component.AsProtoTensorContent(&nonmemcpyable_components.back());
      total_nonmemcpyable_size += nonmemcpyable_components.back().ByteSizeLong();
    }
  }

  // Initialize IOV and allocate space for non-memcpyable data
  Iov iov{element.size() + num_string_tensor_strings - num_string_tensors};
  std::string nonmemcpyable;  // Replaced tstring with std::string for better memory management
  nonmemcpyable.resize(total_nonmemcpyable_size);
  char* nonmemcpyable_pos = nonmemcpyable.data();
  int nonmemcpyable_component_index = 0;

  #pragma omp parallel for  // Parallelizing tensor compression
  for (int i = 0; i < element.size(); ++i) {
    const auto& component = element[i];
    CompressedComponentMetadata* metadata = out->mutable_component_metadata()->Add();
    metadata->set_dtype(component.dtype());
    component.shape().AsProto(metadata->mutable_tensor_shape());

    if (DataTypeCanUseMemcpy(component.dtype())) {
      const TensorBuffer* buffer = DMAHelper::buffer(&component);
      if (buffer) {
        iov.Add(buffer->data(), buffer->size());
        metadata->add_uncompressed_bytes(buffer->size());
      }
    } else if (component.dtype() == DT_STRING) {
      const auto& flats = component.unaligned_flat<tstring>();
      for (int i = 0; i < flats.size(); ++i) {
        iov.Add(const_cast<char*>(flats.data()[i].data()), flats.data()[i].size());
        metadata->add_uncompressed_bytes(flats.data()[i].size());
      }
    } else {
      TensorProto& proto = nonmemcpyable_components[nonmemcpyable_component_index++];
      proto.SerializeToArray(nonmemcpyable_pos, proto.ByteSizeLong());
      iov.Add(nonmemcpyable_pos, proto.ByteSizeLong());
      nonmemcpyable_pos += proto.ByteSizeLong();
      metadata->add_uncompressed_bytes(proto.ByteSizeLong());
    }
  }

  // Ensure we don't exceed Snappy's 4GB limit
  if (iov.NumBytes() > kMaxSnappySize) {
    return errors::OutOfRange("Dataset element size exceeds 4GB Snappy limit: ", iov.NumBytes());
  }

  // Compress using Snappy
  if (!port::Snappy_CompressFromIOVec(iov.Data(), iov.NumBytes(), out->mutable_data())) {
    return errors::Internal("Snappy compression failed.");
  }

  out->set_version(kCompressedElementVersion);
  return absl::OkStatus();
}

Status UncompressElement(const CompressedElement& compressed,
                         std::vector<Tensor>* out) {
  if (compressed.version() != kCompressedElementVersion) {
    return errors::Internal("Unsupported compressed element version: ", compressed.version());
  }

  size_t num_string_tensors = 0;
  size_t num_string_tensor_strings = 0;
  size_t total_nonmemcpyable_size = 0;

  for (const auto& metadata : compressed.component_metadata()) {
    if (metadata.dtype() == DT_STRING) {
      ++num_string_tensors;
      num_string_tensor_strings += metadata.uncompressed_bytes_size();
    } else if (!DataTypeCanUseMemcpy(metadata.dtype())) {
      total_nonmemcpyable_size += metadata.uncompressed_bytes(0);
    }
  }

  Iov iov{compressed.component_metadata_size() + num_string_tensor_strings - num_string_tensors};
  std::string nonmemcpyable;
  nonmemcpyable.resize(total_nonmemcpyable_size);
  char* nonmemcpyable_pos = nonmemcpyable.data();

  for (const auto& metadata : compressed.component_metadata()) {
    if (DataTypeCanUseMemcpy(metadata.dtype())) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      TensorBuffer* buffer = DMAHelper::buffer(&out->back());
      if (buffer) {
        iov.Add(buffer->data(), metadata.uncompressed_bytes(0));
      }
    } else if (metadata.dtype() == DT_STRING) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      const auto& flats = out->back().unaligned_flat<tstring>();
      for (int i = 0; i < metadata.uncompressed_bytes_size(); ++i) {
        flats.data()[i].resize(metadata.uncompressed_bytes(i));
        iov.Add(flats.data()[i].mdata(), metadata.uncompressed_bytes(i));
      }
    } else {
      out->emplace_back();
      iov.Add(nonmemcpyable_pos, metadata.uncompressed_bytes(0));
      nonmemcpyable_pos += metadata.uncompressed_bytes(0);
    }
  }

  const std::string& compressed_data = compressed.data();
  size_t uncompressed_size;
  if (!port::Snappy_GetUncompressedLength(compressed_data.data(), compressed_data.size(), &uncompressed_size)) {
    return errors::Internal("Snappy uncompressed length mismatch. Compressed data size: ", compressed_data.size());
  }

  if (uncompressed_size != static_cast<size_t>(iov.NumBytes())) {
    return errors::Internal("Uncompressed size mismatch: Snappy expects ", uncompressed_size, 
                            " whereas tensor metadata suggests ", iov.NumBytes());
  }

  if (!port::Snappy_UncompressToIOVec(compressed_data.data(), compressed_data.size(), iov.Data(), iov.NumPieces())) {
    return errors::Internal("Snappy decompression failed.");
  }

  return absl::OkStatus();
}

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CompressedElement, "tensorflow.data.CompressedElement");

}  // namespace data
}  // namespace tensorflow
