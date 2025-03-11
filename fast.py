# server.py
import numpy as np
import litserve as ls

# STEP 1: DEFINE YOUR MODEL API
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # Setup the model so it can be called in `predict`.
        # this is a fake model (although a pure math model)... replace it with an LLM or whatever
        self.model = lambda x: x**2

    def decode_request(self, request):
        # Convert the request payload to your model input.
        return request["input"]

    def batch(self, inputs):
        # Convert the list of input requests into a batch
        return np.asarray(inputs)

    def predict(self, x):
        # Run the model on the input and return the output.
        return self.model(x)

    def unbatch(self, outputs):
        # Unbatch the model output and return a list of outputs for each request
        return list(outputs)

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output}


# STEP 2: START THE SERVER
if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, max_batch_size=8, workers_per_device=2, batch_timeout=0.0001)
    server.run(port=8000)