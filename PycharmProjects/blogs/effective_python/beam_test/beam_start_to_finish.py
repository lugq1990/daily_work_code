import apache_beam as beam
# from apache_beam.options.pipeline_options import PipelineOptions

# # costum pipeline
# class CumstomPipeOptions(PipelineOptions):
#     @classmethod
#     def _add_argparse_args(cls, parser) -> None:
#         parser.add_argument("--input-file", default='sample.txt', help='input file for reading beam')
#         parser.add_argument("--output-path", default='output', help='output file for writing beam')
#         return super()._add_argparse_args(parser)

# beam_options = PipelineOptions()


# class ComputeLength(beam.DoFn):
#     def process(self, element, *args, **kwargs):
#         ele_split = element.split(" ")
#         return [len(ele_split)]


# emails_list = [
#     ('amy', 'amy@example.com'),
#     ('carl', 'carl@example.com'),
#     ('julia', 'julia@example.com'),
#     ('carl', 'carl@email.com'),
# ]
# phones_list = [
#     ('amy', '111-222-3333'),
#     ('james', '222-333-4444'),
#     ('amy', '333-444-5555'),
#     ('carl', '444-555-6666'),
# ]

# def join_info(name_info):
#     (name, info) = name_info
#     return '%s, %s, %s' % (name, sorted(info['emails']), sorted(info['phones']))

# with beam.Pipeline() as pipe:
#     # lines = (
#     #     pipe
#     #     | "read" >> beam.io.ReadFromText('./sample.txt')
#     #     # | "length" >> beam.ParDo(ComputeLength())
#     #     # | "length_lambda" >> beam.FlatMap(lambda x: x.split(' '))
#     #     | "split" >> beam.FlatMap(lambda x: x.split(" "))
#     #     | "map" >> beam.Map(lambda x: (x, 1))
#     #     | "combine" >> beam.GroupByKey()
#     #     # | "in memory" >> beam.Create([1, 2, 3])
#     #     | "print" >> beam.Map(print)
#     # )
#     emails = pipe | "emails" >> beam.Create(emails_list)
#     phones = pipe | "phones" >> beam.Create(phones_list)
    
#     res = ({"emails": emails, 'phones': phones} | beam.CoGroupByKey())
    
#     contract_lines = res | beam.Map(join_info)
#     contract_lines | beam.Map(print)

def bound_sum(values, bound=30):
    return min(sum(values ), bound)
    
with beam.Pipeline() as pipe:
    pc = [1,2,32, 49]
    small_sum = pc | beam.CombineGlobally(bound_sum)
    large_sum = pc | beam.CombineGlobally(bound_sum, 40)
    
    merged_sum = (small_sum, large_sum) | beam.Flatten()
    
    merged_sum | beam.Map(print)
    
print("⭐⭐⭐⭐⭐ Finish ⭐⭐⭐⭐⭐")