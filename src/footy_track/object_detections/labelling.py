import labelbox as lb

from footy_track.object_detections.constants import PERSON_TAG
from footy_track.object_detections.detectors import BALL_TAG



ONTOLOGY_NAME = "Simplified Detection Ontology"
COMPLEX_ONTOLOGY_ID = "cmg9emo7y00hr072h5i0y5e9o"   
SIMPLIFIED_ONTOLOGY_ID = "cmigor8a206ub07y58twc4o4i"


SIMPLIFIED_ONTOLOGY = {
    "name": ONTOLOGY_NAME,
    "tools": [
        {"tool": "rectangle", "name": BALL_TAG},
        {"tool": "rectangle", "name": PERSON_TAG},
    ],
    "classifications": [
        {
            "name": "broadcast_frame",
            "instructions": "Broadcast frame",
            "type": "radio",
            "options": [
                {"label": "yes", "value": "yes"},
                {"label": "no", "value": "no"}
            ]
        }
    ]
}

def upload_simplified_ontology(client):
    """
    Upload the simplified ontology to Labelbox using a Client instance.
    If the Labelbox client provides a `create_ontology` method it will be used;
    otherwise this function returns the local ontology dict.
    """
    if hasattr(client, "create_ontology"):
        return client.create_ontology(SIMPLIFIED_ONTOLOGY)
    return SIMPLIFIED_ONTOLOGY

class SimplifiedOntology:
    """
    Encapsulates the simplified Labelbox ontology and provides helpers to
    create or retrieve an existing ontology from a Labelbox Client.
    """

    def __init__(self, id: str | None = None):
        self.ontology_data = SIMPLIFIED_ONTOLOGY
        self.name = self.ontology_data.get("name")
        self.id = id
        self.client = lb.Client()

        if not id:
            self.ontology = self.get_ontology()


    def to_dict(self):
        return self.ontology

    def get_ontology(self):
        """Retrieve the ontology using the client's get_ontology if available."""
        if self.id is None:
            raise RuntimeError("Ontology ID is not set. Create it or get a new one")
        ontology = self.client.get_ontology(self.id)
        self.ontology = ontology
        return ontology

    def create(self):
        """Create the ontology using the client's create_ontology if available."""
        if self.client is None:
            raise RuntimeError("Labelbox client is required to create an ontology.")
        # id = self.client.create_ontology(self.ontology)
        ontology = self.client.create_ontology(
            name=self.ontology_data["name"],
            normalized={
                "tools": self.ontology_data["tools"],
                "classifications": self.ontology_data["classifications"],
            },
        )
        self.ontology = ontology
        self.id = ontology.uid
        return ontology


