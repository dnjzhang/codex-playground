class LoadProperties:

    def __init__(self):
        import json
        # reading the data from the file
        with open('oci-config.json') as f:
            data = f.read()

        js = json.loads(data)

        self.model_name = js["model_name"]
        self.endpoint = js["endpoint"]
        self.compartment_ocid = js["compartment_ocid"]
        self.embedding_model_name = js["embedding_model_name"]
        self.config_profile = js["config_profile"]

    def getModelName(self):
        return self.model_name

    def getEndpoint(self):
        return self.endpoint

    def getCompartment(self):
        return self.compartment_ocid

    def getConfigProfile(self):
        return self.config_profile

    def getEmbeddingModelName(self):
        return self.embedding_model_name

    def getLangChainKey(self):
        return self.langchain_key

    def getlangChainEndpoint(self):
        return self.langchain_endpoint
