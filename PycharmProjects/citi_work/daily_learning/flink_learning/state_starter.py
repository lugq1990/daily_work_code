from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, FlatMapFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig
from pyflink.common.time import Time


class CountWindow(FlatMapFunction):
    def __init__(self) -> None:
        self.sum = None
    
    def open(self, runtime_context: RuntimeContext):
        desc = ValueStateDescriptor('avg', Types.PICKLED_BYTE_ARRAY())
        self.sum = runtime_context.get_state(desc)
        
    def flat_map(self, value):
        cur_sum = self.sum.value()
        if  cur_sum is None:
            cur_sum = (0, 0)
        cur_sum = (cur_sum[0] + 1, cur_sum[1] + value[1])
        self.sum.update(cur_sum)
        
        if cur_sum[0] >= 2:
            self.sum.clear()
            yield value[0], int(cur_sum[1] / cur_sum[0])
            

state = ValueStateDescriptor('test', Types.STRING())
ttl = StateTtlConfig.new_builder(Time.milliseconds(1)).disable_cleanup_in_background()\
    .cleanup_incrementally()\
    .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)\
        .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired).build()
            

env = StreamExecutionEnvironment.get_execution_environment()

env.set_default_local_parallelism(1)
env.get_checkpoint_config().

ds = env.from_collection([(1, 3), (1, 5), (1, 7), (1, 4), (1, 2)]).key_by(lambda x: x[0]).flat_map(CountWindow()).print()
env.execute()