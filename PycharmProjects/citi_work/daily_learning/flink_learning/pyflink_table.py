from pyflink.table import TableEnvironment, EnvironmentSettings, TableDescriptor, Schema, DataTypes, FormatDescriptor
from pyflink.common import Row
from pyflink.table.expressions import lit, col 
from pyflink.table.udf import udtf 
import os

word_count_data = ["To be, or not to be,--that is the question:--",
                   "Whether 'tis nobler in the mind to suffer",
                   "The slings and arrows of outrageous fortune",
                   "Or to take arms against a sea of troubles,",
                   "And by opposing end them?--To die,--to sleep,--",
                   "No more; and by a sleep to say we end",
                   "The heartache, and the thousand natural shocks",
                   "That flesh is heir to,--'tis a consummation",
                   "Devoutly to be wish'd. To die,--to sleep;--",
                   "To sleep! perchance to dream:--ay, there's the rub;",
                   "For in that sleep of death what dreams may come,",
                   "When we have shuffled off this mortal coil,",
                   "Must give us pause: there's the respect",
                   "That makes calamity of so long life;",
                   "For who would bear the whips and scorns of time,",
                   "The oppressor's wrong, the proud man's contumely,",
                   "The pangs of despis'd love, the law's delay,",
                   "The insolence of office, and the spurns",
                   "That patient merit of the unworthy takes,",
                   "When he himself might his quietus make",
                   "With a bare bodkin? who would these fardels bear,",
                   "To grunt and sweat under a weary life,",
                   "But that the dread of something after death,--",
                   "The undiscover'd country, from whose bourn",
                   "No traveller returns,--puzzles the will,",
                   "And makes us rather bear those ills we have",
                   "Than fly to others that we know not of?",
                   "Thus conscience does make cowards of us all;",
                   "And thus the native hue of resolution",
                   "Is sicklied o'er with the pale cast of thought;",
                   "And enterprises of great pith and moment,",
                   "With this regard, their currents turn awry,",
                   "And lose the name of action.--Soft you now!",
                   "The fair Ophelia!--Nymph, in thy orisons",
                   "Be all my sins remember'd."]

env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())

env.get_config().set("parallelism.default", "1")

input_path = os.path.dirname(os.path.abspath(__file__)) 
input_path = os.path.join(input_path,'test.csv')

print(input_path)


env.create_temporary_table('test', 
                           TableDescriptor.for_connector('filesystem')
                           .schema(Schema.new_builder().column('word', DataTypes.STRING()).build())
                           .option('path', input_path)
                           .format('csv')
                           .build())

tab = env.from_path('test')

# out
env.create_temporary_table('out', 
                           TableDescriptor.for_connector('print')
                           .schema(Schema.new_builder().column('word', DataTypes.STRING()).column('count', DataTypes.BIGINT()).build()).build())

@udtf(result_types=[DataTypes.STRING()])
def split(x: Row):
    for t in x[0].split():
        yield Row(t)
        
tab.flat_map(split).alias('word').group_by(col('word')).select(col('word'), lit(1).count).execute_insert('out').wait()

print(tab.explain())