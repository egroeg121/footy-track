import labelbox as lb

ONTOLOGY_NAME = "Game Ontology"
ONTOLOGY_ID = "cmg9emo7y00hr072h5i0y5e9o"


def get_ontology() -> lb.Ontology:
    """Get the ontology for the project"""
    client = lb.Client()
    return client.get_ontology(ONTOLOGY_ID)


# def preload_
