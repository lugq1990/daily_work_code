import apache_beam as beam
from apache_beam.transforms.window import TimestampCombiner, Sessions, Duration, TimestampedValue
from apache_beam.io.textio import WriteToText
from apache_beam.utils import timestamp


class AddTimestamp(beam.DoFn):
    def process(self, element, *args, **kwargs):
        time = element['timestamp']
        element = (element['userId'], element['click'])
        
        yield TimestampedValue(element, time)


with beam.Pipeline() as pipe:
    events = pipe | beam.Create(
        [
            {"userId": "Andy", "click": 1, "timestamp": 1603112520},  # Event time: 13:02
            {"userId": "Sam", "click": 1, "timestamp": 1603113240},  # Event time: 13:14
            {"userId": "Andy", "click": 1, "timestamp": 1603115820},  # Event time: 13:57
            {"userId": "Andy", "click": 1, "timestamp": 1603113600},  # Event time: 13:20
        ]
    )
    
    timestamp_events = events |"add timestamp" >> beam.ParDo(AddTimestamp())
    windowed_events = timestamp_events | "based on window" >> beam.WindowInto(Sessions(gap_size=30*60), trigger=None, accumulation_mode=None, timestamp_combiner=None, allowed_lateness=Duration(seconds=1 * 24 * 60 *60)) 
    
    sum_click = windowed_events | beam.CombinePerKey(sum)
    
    sum_click | WriteToText(file_path_prefix='output', file_name_suffix='.txt')