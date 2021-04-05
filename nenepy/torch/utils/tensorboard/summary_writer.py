import os
import socket
import time

from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.event_pb2 import SessionLog
from tensorboard.summary.writer.event_file_writer import EventFileWriter as TFEventFileWriter
from tensorboard.summary.writer.event_file_writer import _AsyncWriter
from tensorboard.summary.writer.record_writer import RecordWriter
from torch.utils.tensorboard import SummaryWriter as TFSummaryWriter
from torch.utils.tensorboard import FileWriter as TFFileWriter

#
# class EventFileWriter(TFEventFileWriter):
#
#     def __init__(self, logdir, max_queue_size=10, flush_secs=120, filename_suffix=""):
#         super(EventFileWriter, self).__init__()
#         self._logdir = logdir
#         if not tf.io.gfile.exists(logdir):
#             tf.io.gfile.makedirs(logdir)
#         self._file_name = (
#                 os.path.join(
#                     logdir,
#                     "events.out.tfevents.%010d.%s.%s.%s"
#                     % (
#                         time.time(),
#                         socket.gethostname(),
#                         os.getpid(),
#                         _global_uid.get(),
#                     ),
#                 )
#                 + filename_suffix
#         )  # noqa E128
#         self._general_file_writer = tf.io.gfile.GFile(self._file_name, "wb")
#         self._async_writer = _AsyncWriter(
#             RecordWriter(self._general_file_writer), max_queue_size, flush_secs
#         )
#
#         # Initialize an event instance.
#         _event = event_pb2.Event(
#             wall_time=time.time(), file_version="brain.Event:2"
#         )
#         self.add_event(_event)
#         self.flush()
#
#
# class FileWriter(TFFileWriter):
#
#     def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix=''):
#         super(FileWriter, self).__init__(log_dir, max_queue, flush_secs, filename_suffix)
#         log_dir = str(log_dir)
#         self.event_writer = EventFileWriter(log_dir, max_queue, flush_secs, filename_suffix)

#
# class SummaryWriter(TFSummaryWriter):
#
#     # ==================================================================================================
#     #
#     #   Override Method
#     #
#     # ==================================================================================================
#     def _get_file_writer(self):
#         """Returns the default FileWriter instance. Recreates it if closed."""
#         if self.all_writers is None or self.file_writer is None:
#             self.file_writer = FileWriter(self.log_dir, self.max_queue,
#                                           self.flush_secs, self.filename_suffix)
#             self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
#             if self.purge_step is not None:
#                 most_recent_step = self.purge_step
#                 self.file_writer.add_event(
#                     Event(step=most_recent_step, file_version='brain.Event:2'))
#                 self.file_writer.add_event(
#                     Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
#                 self.purge_step = None
#         return self.file_writer
