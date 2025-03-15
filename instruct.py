import os
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional
import json

from pydantic import BaseModel, Field
from mistralai import Mistral
import yaml

"""
This file will assist you in creating the codebase and help you understand the flow of elements. The overall objective is to setup a hybrid RAG pipeline
utilizing the following in the order:

- MistralAI for pdf -> text, signed_url
- Google Embedding -> text -> embedding
- pgvector -> embedding storage
- redis -> cache
- gemini flash -> querying and retrieval
- docker - > containerization of the pipeline and testing
- config.yaml -> configuration file for the pipeline

Now, onto the various code snippets
"""

#First -> the mistral ocr 
from mistralai import Mistral

class MistralProcessor:
    def __init__(self):
        self.client = Mistral(api_key="your_api_key")
        self.file_id = None
        self.signed_url = None
    
    def upload_pdf(self, pdf_path):
        with open(pdf_path, "rb") as f:
            uploaded_pdf = self.client.files.upload(
                file={
                    "file_name": Path(pdf_path).name,
                    "content": f,
                },
                purpose="ocr"
            )
        self.file_id = uploaded_pdf.id
        return self.file_id

    def get_signed_url(self):
        self.client.files.retrieve(file_id=self.file_id)
        self.signed_url = self.client.files.get_signed_url(file_id=self.file_id)
        return self.signed_url.url

    def process_ocr(self):
        url = self.get_signed_url()
        ocr_response = self.client.ocr.process(
            model="<OCR_MODEL>",
            document={
                "type": "document_url",
                "document_url": url,
            }
        )
        return ocr_response

#Second -> the google embedding
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?")

#Note that this embedding model has a token size of 8k. So we need to chunk the text into 8k tokens and then embed them

#Third -> the pgvector
"""
Setup directions and usage guide:
%docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
 Add items to vector store

Note that adding documents by ID will over-write any existing documents that match that ID.

docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   use_jsonb=True,
)

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

vector_store.delete(ids=["3"])

Filtering Support

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.
Operator	Meaning/Category
$eq	Equality (==)
$ne	Inequality (!=)
$lt	Less than (<)
$lte	Less than or equal (<=)
$gt	Greater than (>)
$gte	Greater than or equal (>=)
$in	Special Cased (in)
$nin	Special Cased (not in)
$between	Special Cased (between)
$like	Text (like)
$ilike	Text (case-insensitive like)
$and	Logical (and)
$or	Logical (or)

results = vector_store.similarity_search(
    "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

* there are cats in the pond [{'id': 1, 'topic': 'animals', 'location': 'pond'}]
* the library hosts a weekly story time for kids [{'id': 9, 'topic': 'reading', 'location': 'library'}]
* ducks are also found in the pond [{'id': 2, 'topic': 'animals', 'location': 'pond'}]
* the new art exhibit is fascinating [{'id': 5, 'topic': 'art', 'location': 'museum'}]

vector_store.similarity_search(
    "ducks",
    k=10,
    filter={"id": {"$in": [1, 5, 2, 9]}, "location": {"$in": ["pond", "market"]}},
)

[Document(metadata={'id': 1, 'topic': 'animals', 'location': 'pond'}, page_content='there are cats in the pond'),
 Document(metadata={'id': 2, 'topic': 'animals', 'location': 'pond'}, page_content='ducks are also found in the pond')]

 vector_store.similarity_search(
    "ducks",
    k=10,
    filter={
        "$and": [
            {"id": {"$in": [1, 5, 2, 9]}},
            {"location": {"$in": ["pond", "market"]}},
        ]
    },
)

[Document(metadata={'id': 1, 'topic': 'animals', 'location': 'pond'}, page_content='there are cats in the pond'),
 Document(metadata={'id': 2, 'topic': 'animals', 'location': 'pond'}, page_content='ducks are also found in the pond')]

results = vector_store.similarity_search_with_score(query="cats", k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

* [SIM=0.763449] there are cats in the pond [{'id': 1, 'topic': 'animals', 'location': 'pond'}]

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("kitty")

[Document(metadata={'id': 1, 'topic': 'animals', 'location': 'pond'}, page_content='there are cats in the pond')]
"""

#Fourth -> the redis cache
"""
# connection to redis standalone at localhost, db 0, no password
redis_url = "redis://localhost:6379"
# connection to host "redis" port 7379 with db 2 and password "secret" (old style authentication scheme without username / pre 6.x)
redis_url = "redis://:secret@redis:7379/2"
# connection to host redis on default port with user "joe", pass "secret" using redis version 6+ ACLs
redis_url = "redis://joe:secret@redis/0"

# connection to sentinel at localhost with default group mymaster and db 0, no password
redis_url = "redis+sentinel://localhost:26379"
# connection to sentinel at host redis with default port 26379 and user "joe" with password "secret" with default group mymaster and db 0
redis_url = "redis+sentinel://joe:secret@redis"
# connection to sentinel, no auth with sentinel monitoring group "zone-1" and database 2
redis_url = "redis+sentinel://redis:26379/zone-1/2"

# connection to redis standalone at localhost, db 0, no password but with TLS support
redis_url = "rediss://localhost:6379"
# connection to redis sentinel at localhost and default port, db 0, no password
# but with TLS support for booth Sentinel and Redis server
redis_url = "rediss+sentinel://localhost"

docker run -d -p 6379:6379 redis/redis-stack:latest

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
Connecting to Redis at: redis://redis:6379

Sample Data

The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics. We'll use a subset for this demonstration and focus on two categories: 'alt.atheism' and 'sci.space':

from langchain.docstore.document import Document
from sklearn.datasets import fetch_20newsgroups

categories = ["alt.atheism", "sci.space"]
newsgroups = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

# Use only the first 250 documents
texts = newsgroups.data[:250]
metadata = [
    {"category": newsgroups.target_names[target]} for target in newsgroups.target[:250]
]

len(texts)
250

from langchain_redis import RedisConfig, RedisVectorStore

config = RedisConfig(
    index_name="newsgroups",
    redis_url=REDIS_URL,
    metadata_schema=[
        {"name": "category", "type": "tag"},
    ],
)

vector_store = RedisVectorStore(embeddings, config=config)
ids = vector_store.add_texts(texts, metadata)

print(ids[0:10])
['newsgroups:f1e788ee61fe410daa8ef941dd166223', 'newsgroups:80b39032181f4299a359a9aaed6e2401', 'newsgroups:99a3efc1883647afba53d115b49e6e92', 'newsgroups:503a6c07cd71418eb71e11b42589efd7', 'newsgroups:7351210e32d1427bbb3c7426cf93a44f', 'newsgroups:4e79fdf67abe471b8ee98ba0e8a1a055', 'newsgroups:03559a1d574e4f9ca0479d7b3891402e', 'newsgroups:9a1c2a7879b8409a805db72feac03580', 'newsgroups:3578a1e129f5435f9743cf803413f37a', 'newsgroups:9f68baf4d6b04f1683d6b871ce8ad92d']

texts[0], metadata[0]
('From: bil@okcforum.osrhe.edu (Bill Conner)\nSubject: Re: Not the Omni!\nNntp-Posting-Host: okcforum.osrhe.edu\nOrganization: Okcforum Unix Users Group\nX-Newsreader: TIN [version 1.1 PL6]\nLines: 18\n\nCharley Wingate (mangoe@cs.umd.edu) wrote:\n: \n: >> Please enlighten me.  How is omnipotence contradictory?\n: \n: >By definition, all that can occur in the universe is governed by the rules\n: >of nature. Thus god cannot break them. Anything that god does must be allowed\n: >in the rules somewhere. Therefore, omnipotence CANNOT exist! It contradicts\n: >the rules of nature.\n: \n: Obviously, an omnipotent god can change the rules.\n\nWhen you say, "By definition", what exactly is being defined;\ncertainly not omnipotence. You seem to be saying that the "rules of\nnature" are pre-existant somehow, that they not only define nature but\nactually cause it. If that\'s what you mean I\'d like to hear your\nfurther thoughts on the question.\n\nBill\n',
 {'category': 'alt.atheism'})

# Delete documents by passing one or more keys/ids
vector_store.index.drop_keys(ids[0])
1
# assumes you're running Redis locally (use --host, --port, --password, --username, to change this)
!rvl index listall --port 6379
[32m17:54:50[0m [34m[RedisVL][0m [1;30mINFO[0m   Using Redis address from environment variable, REDIS_URL
[32m17:54:50[0m [34m[RedisVL][0m [1;30mINFO[0m   Indices:
[32m17:54:50[0m [34m[RedisVL][0m [1;30mINFO[0m   1. newsgroups
!rvl index info -i newsgroups --port 6379
query = "Tell me about space exploration"
results = vector_store.similarity_search(query, k=2)

print("Simple Similarity Search Results:")
for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()

Simple Similarity Search Results:
Content: From: aa429@freenet.carleton.ca (Terry Ford)
Subject: A flawed propulsion system: Space Shuttle
X-Ad...
Metadata: {'category': 'sci.space'}

Content: From: nsmca@aurora.alaska.edu
Subject: Space Design Movies?
Article-I.D.: aurora.1993Apr23.124722.1
...
Metadata: {'category': 'sci.space'}
# Similarity search with score and filter
scored_results = vector_store.similarity_search_with_score(query, k=2)

print("Similarity Search with Score Results:")
for doc, score in scored_results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print()
Similarity Search with Score Results:
Content: From: aa429@freenet.carleton.ca (Terry Ford)
Subject: A flawed propulsion system: Space Shuttle
X-Ad...
Metadata: {'category': 'sci.space'}
Score: 0.569670975208

Content: From: nsmca@aurora.alaska.edu
Subject: Space Design Movies?
Article-I.D.: aurora.1993Apr23.124722.1
...
Metadata: {'category': 'sci.space'}
Score: 0.590400338173

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever.invoke("What planet in the solar system has the largest number of moons?")
[Document(metadata={'category': 'sci.space'}, page_content='Subject: Re: Comet in Temporary Orbit Around Jupiter?\nFrom: Robert Coe <bob@1776.COM>\nDistribution: world\nOrganization: 1776 Enterprises, Sudbury MA\nLines: 23\n\njgarland@kean.ucs.mun.ca writes:\n\n> >> Also, perihelions of Gehrels3 were:\n> >> \n> >> April  1973     83 jupiter radii\n> >> August 1970     ~3 jupiter radii\n> > \n> > Where 1 Jupiter radius = 71,000 km = 44,000 mi = 0.0005 AU.  So the\n> > 1970 figure seems unlikely to actually be anything but a perijove.\n> > Is that the case for the 1973 figure as well?\n> > -- \n> Sorry, _perijoves_...I\'m not used to talking this language.\n\nHmmmm....  The prefix "peri-" is Greek, not Latin, so it\'s usually used\nwith the Greek form of the name of the body being orbited.  (That\'s why\nit\'s "perihelion" rather than "perisol", "perigee" rather than "periterr",\nand "pericynthion" rather than "perilune".)  So for Jupiter I\'d expect it\nto be something like "perizeon".)   :^)\n\n   ___            _                                           -  Bob\n   /__) _   /    / ) _   _\n(_/__) (_)_(_)  (___(_)_(/_______________________________________ bob@1776.COM\nRobert K. Coe ** 14 Churchill St, Sudbury, Massachusetts 01776 ** 508-443-3265\n'),
 Document(metadata={'category': 'sci.space'}, page_content='From: pyron@skndiv.dseg.ti.com (Dillon Pyron)\nSubject: Re: Why not give $1 billion to first year-long moon residents?\nLines: 42\nNntp-Posting-Host: skndiv.dseg.ti.com\nReply-To: pyron@skndiv.dseg.ti.com\nOrganization: TI/DSEG VAX Support\n\n\nIn article <1qve4kINNpas@sal-sun121.usc.edu>, schaefer@sal-sun121.usc.edu (Peter Schaefer) writes:\n>In article <1993Apr19.130503.1@aurora.alaska.edu>, nsmca@aurora.alaska.edu writes:\n>|> In article <6ZV82B2w165w@theporch.raider.net>, gene@theporch.raider.net (Gene Wright) writes:\n>|> > With the continuin talk about the "End of the Space Age" and complaints \n>|> > by government over the large cost, why not try something I read about \n>|> > that might just work.\n>|> > \n>|> > Announce that a reward of $1 billion would go to the first corporation \n>|> > who successfully keeps at least 1 person alive on the moon for a year. \n>|> > Then you\'d see some of the inexpensive but not popular technologies begin \n>|> > to be developed. THere\'d be a different kind of space race then!\n>|> > \n>|> > --\n>|> >   gene@theporch.raider.net (Gene Wright)\n>|> > theporch.raider.net  615/297-7951 The MacInteresteds of Nashville\n>|> ====\n>|> If that were true, I\'d go for it.. I have a few friends who we could pool our\n>|> resources and do it.. Maybe make it a prize kind of liek the "Solar Car Race"\n>|> in Australia..\n>|> Anybody game for a contest!\n>|> \n>|> ==\n>|> Michael Adams, nsmca@acad3.alaska.edu -- I\'m not high, just jacked\n>\n>\n>Oh gee, a billion dollars!  That\'d be just about enough to cover the cost of the\n>feasability study!  Happy, Happy, JOY! JOY!\n>\n\nFeasability study??  What a wimp!!  While you are studying, others would be\ndoing.  Too damn many engineers doing way too little engineering.\n\n"He who sits on his arse sits on his fortune"  - Sir Richard Francis Burton\n--\nDillon Pyron                      | The opinions expressed are those of the\nTI/DSEG Lewisville VAX Support    | sender unless otherwise stated.\n(214)462-3556 (when I\'m here)     |\n(214)492-4656 (when I\'m home)     |Texans: Vote NO on Robin Hood.  We need\npyron@skndiv.dseg.ti.com          |solutions, not gestures.\nPADI DM-54909                     |\n\n')]

"""

#Fifth -> the gemini flash
"""
# Create a new flash instance   
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg
"""

#Sixth -> the docker containerization