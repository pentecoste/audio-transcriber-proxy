# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: audiotranscription.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'audiotranscription.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18\x61udiotranscription.proto\x1a\x1bgoogle/protobuf/empty.proto\"#\n\x0c\x41udioRequest\x12\x13\n\x0b\x62\x61se64_data\x18\x01 \x01(\t\"\x1d\n\x0cTextResponse\x12\r\n\x05value\x18\x01 \x01(\t2\xd3\x01\n\x19\x41udioTranscriptionService\x12-\n\rprocess_audio\x12\r.AudioRequest\x1a\r.TextResponse\x12\x42\n\x10stop_transcriber\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\x12\x43\n\x11start_transcriber\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Emptyb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'audiotranscription_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_AUDIOREQUEST']._serialized_start=57
  _globals['_AUDIOREQUEST']._serialized_end=92
  _globals['_TEXTRESPONSE']._serialized_start=94
  _globals['_TEXTRESPONSE']._serialized_end=123
  _globals['_AUDIOTRANSCRIPTIONSERVICE']._serialized_start=126
  _globals['_AUDIOTRANSCRIPTIONSERVICE']._serialized_end=337
# @@protoc_insertion_point(module_scope)
