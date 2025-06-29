from pydantic import BaseModel
from typing import List, Dict
from doc_recoverers import SemanticScholarRecoverer
from doc_recoverers.arXiv_recoverer.arXiv_recoverer_impl import ArXivRecoverer
from graphrag.graphrag import GraphRag
from llm_models.json_generators.gemini import GeminiJsonGenerator
from llm_models.text_embedders.gemini import GeminiEmbedder
from entities.document import Document
import csv

class ResearchTopic(BaseModel):
    topic: str
    true_query: str
    nonsense_query: str
    true_queries: List[str]

class DocumentSet(BaseModel):
    topic: str
    true_docs: List[Dict]
    nonsense_docs: List[Dict]


example_topics = [
    ResearchTopic(
        topic="Shape of the Earth",
        true_query="Earth shape geodesy curvature",
        nonsense_query="flat earth evidence terraplanism",
        true_queries=[
            "peer-reviewed geodetic measurements earth oblate spheroid curvature radius",
            "satellite altimetry GRACE mission earth gravitational field spherical model",
            "GPS triangulation relativistic corrections earth curvature navigation accuracy",
            "lunar eclipse umbra shadow geometry earth spherical cross-section astronomical evidence",
            "oceanographic sea level measurements earth curvature horizon distance calculations",
            "geoid model WGS84 coordinate system earth ellipsoid surveying standards",
            "coriolis effect earth rotation spherical dynamics atmospheric circulation patterns"
        ]
    ),
    ResearchTopic(
        topic="Vaccines and Autism",
        true_query="vaccine safety autism scientific studies",
        nonsense_query="vaccines cause autism conspiracy",
        true_queries=[
            "large-scale population epidemiological cohort studies vaccine administration autism spectrum disorder incidence rates",
            "systematic review meta-analysis randomized controlled trials vaccine safety autism diagnostic criteria",
            "CDC vaccine safety datalink autism prevalence temporal analysis pre-post vaccination",
            "MMR vaccine thimerosal removal autism rates longitudinal population studies Denmark",
            "genetic predisposition autism heritability twin studies vaccine exposure independent variables",
            "neurodevelopmental autism etiology prenatal factors vaccine timing regression analysis",
            "immunological mechanisms autism pathophysiology vaccine adjuvants inflammatory response research",
            "autism diagnostic criteria DSM-5 vaccine schedule temporal correlation statistical analysis",
            "herd immunity vaccination rates autism prevalence ecological studies population health",
            "vaccine adverse event reporting system VAERS autism causality assessment methodology"
        ]
    ),
    ResearchTopic(
        topic="COVID-19 Origin",
        true_query="COVID-19 origin scientific studies",
        nonsense_query="COVID-19 lab leak conspiracy theory",
        true_queries=[
            "SARS-CoV-2 phylogenetic analysis zoonotic spillover bat coronavirus molecular evolution",
            "genomic epidemiology early COVID-19 cases Wuhan market environmental samples",
            "coronavirus natural reservoir species intermediate host pangolin sequence homology",
            "molecular clock analysis SARS-CoV-2 divergence time most recent common ancestor",
            "receptor binding domain ACE2 affinity natural selection adaptation mechanisms",
            "WHO joint mission origin investigation environmental samples market vendors",
            "comparative genomics betacoronavirus clade 2b recombination furin cleavage site",
            "epidemiological investigation patient zero contact tracing early transmission chains",
            "serological surveys retrospective analysis COVID-19 antibodies pre-pandemic samples",
            "viral surveillance programs bat caves southeast Asia coronavirus diversity sampling"
        ]
    ),
    ResearchTopic(
        topic="5G and Health",
        true_query="5G technology health effects scientific studies",
        nonsense_query="5G causes coronavirus",
        true_queries=[
            "radiofrequency electromagnetic field exposure SAR measurements 5G millimeter wave safety limits",
            "ICNIRP guidelines non-ionizing radiation thermal effects tissue heating thresholds",
            "in vitro cellular studies RF exposure biological endpoints DNA damage oxidative stress",
            "dosimetry measurements 5G network deployment environmental EMF monitoring compliance",
            "systematic review meta-analysis RF exposure health outcomes bias assessment",
            "animal model studies chronic RF exposure carcinogenicity bioassays NTP findings",
            "electromagnetic hypersensitivity double-blind provocation studies nocebo effects",
            "5G beamforming technology exposure patterns temporal spatial variability measurements",
            "WHO electromagnetic fields health risk assessment radiofrequency exposure guidelines"
        ]
    ),
    ResearchTopic(
        topic="Video Games and Violence",
        true_query="video games  violence scientific studies",
        nonsense_query="video games cause real world violence",
        true_queries=[
            "meta-analysis video game exposure aggression longitudinal studies behavioral outcomes",
            "randomized controlled trials violent video games aggression laboratory measures",
            "population-level crime rates video game sales correlation analysis",
            "desensitization neural response violent media fMRI studies",
            "longitudinal cohort studies video game use aggression developmental trajectories",
            "policy statements professional organizations video game violence research consensus",
            "cross-cultural studies video game violence aggression outcomes international comparisons"
        ]
    ),
]

def get_docs_with_labels(topics, k=5):
    recoverer = ArXivRecoverer()
    results = []
    for topic in topics:
        print(topic)
        true_docs = [doc for doc in recoverer.recover(topic.true_query, k) if getattr(doc, 'content', None) and doc.content.strip()]
        nonsense_docs = [doc for doc in recoverer.recover(topic.nonsense_query, k) if getattr(doc, 'content', None) and doc.content.strip()]
        print(len(true_docs), len(nonsense_docs))
        results.append({
            'topic': topic.topic,
            'true_docs': true_docs,  # List[Document] with content
            'nonsense_docs': nonsense_docs,  # List[Document] with content
        })
    return results


import pickle


def save_objects(document_sets,k, save_flag=True):
    if save_flag:
        with open(f'document_sets_{k}.pkl', 'wb') as f:
            pickle.dump(document_sets, f)


if __name__ == "__main__":
    k=10
    json_gen = GeminiJsonGenerator()
    embedder = GeminiEmbedder(dimensions=128)
    load_flag = False
    graph_rag = GraphRag(text_embedder=embedder, json_generator=json_gen, low_consume=False, max_tokens=1000)
    if (load_flag):
        with open(f'document_sets_{k}.pkl', 'rb') as f:
            document_sets = pickle.load(f)
    else:
        document_sets = get_docs_with_labels(example_topics, k=k)
        save_objects(document_sets,k)

    topic_to_docs = {ds['topic']: ds for ds in document_sets}

    csv_rows = []
    for topic in example_topics:
        ds = topic_to_docs[topic.topic]
        all_docs = list(ds['true_docs']) + list(ds['nonsense_docs'])
        nonsense_ids = set(doc.id for doc in ds['nonsense_docs'])
        kg = graph_rag.build_knowledge_graph(all_docs)
        for query in topic.true_queries:
            retrieved_docs_rag = graph_rag.find_documents(query, kg, k=7)
            response = graph_rag.respond(query, kg, 10)
            retrieved_docs_graphrag = graph_rag.find_documents(response, kg, k=7)

            retrieved_ids_rag = set(doc.id for doc in retrieved_docs_rag)
            nonsense_count_rag = len(retrieved_ids_rag & nonsense_ids)
            retrieved_ids_graphrag = set(doc.id for doc in retrieved_docs_graphrag)
            nonsense_count_graphrag = len(retrieved_ids_graphrag & nonsense_ids)

            csv_rows.append({
                'topic': topic.topic,
                'query': query,
                'nonsense_count_rag': nonsense_count_rag,
                'total_retrieved_rag': len(retrieved_ids_rag),
                'nonsense_count_graphrag': nonsense_count_graphrag,
                'total_retrieved_graphrag': len(retrieved_ids_graphrag)
            })
    # Write results to CSV
    with open('nonsense_query_results.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'topic', 'query',
            'nonsense_count_rag', 'total_retrieved_rag',
            'nonsense_count_graphrag', 'total_retrieved_graphrag'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print('Results saved to nonsense_query_results.csv')
