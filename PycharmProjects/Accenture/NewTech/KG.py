# -*- coding:utf-8 -*-
from neo4j.v1 import GraphDatabase

uri = 'bolt:loalhost:7687'
driver = GraphDatabase.driver(uri, auth=('neo4j', '123lug'))

def submit(sent):
    with driver.session() as sess:
        with sess.begin_transaction() as tx:
            tx.run(sent)

cypher = """
match(x:Person)-[:LOVES]-(y) where x.name = 'lugq' return x, y
"""

# start to commit
submit(cypher)
print('Finished submit jobs!')