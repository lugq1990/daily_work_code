from pyflink.common import Row, WatermarkStrategy, Encoder, Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext, RuntimeExecutionMode
from pyflink.datastream.connectors.file_system import FileSource, FileSink, StreamFormat, OutputFileConfig, RollingPolicy
import os

env = StreamExecutionEnvironment.get_execution_environment()

env.set_default_local_parallelism(1)
env.set_runtime_mode(RuntimeExecutionMode.BATCH)

input_path = os.path.dirname(os.path.abspath(__file__)) 
out_path = input_path
input_path = os.path.join(input_path,'test.txt')

ds = env.from_source(source=FileSource.for_record_stream_format(StreamFormat.text_line_format(), input_path)
                     .process_static_file_set().build(),
                     watermark_strategy=WatermarkStrategy.for_monotonous_timestamps(),
                     source_name='file_source')

def spilt(x):
    yield from x.split()
    
ds = ds.flat_map(spilt)\
    .map(lambda x: (x, 1), 
         output_type=Types.TUPLE([Types.STRING(), Types.INT()]))\
             .key_by(lambda x: x[0])\
                 .reduce(lambda x, y:(x[0], x[1] + y[1]))
    
ds.sink_to(sink=FileSink.for_row_format(base_path=out_path, encoder=Encoder.simple_string_encoder()).with_output_file_config(
    output_file_config=OutputFileConfig.builder().with_part_prefix('res').with_part_suffix('.txt').build()
).with_rolling_policy(RollingPolicy.default_rolling_policy()).build())

env.execute()