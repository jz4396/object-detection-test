#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include "tf_utils.hpp"
//#include <Tensorflow_1-12.h>
#include <opencv2/core.hpp>

using namespace std;



/*
static void DeallocateBuffer(void* data, size_t) {
	free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* path) {
	const auto f = fopen(path, "rb");
	if (f == nullptr) {
		return nullptr;
	}

	fseek(f, 0, SEEK_END);
	const auto fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	if (fsize < 1) {
		fclose(f);
		return nullptr;
	}

	const auto data = std::malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = DeallocateBuffer;

	return buf;
}
*/

using namespace tf_utils;

int main(int argc, char **argv) {
	cout << "Hello from TensorFlow C library version " << TF_Version() << "\n\n";
	
	const char* GraphPath;
	if (argv[2] == NULL) GraphPath = "demo/ssd_mobilenet_v1_egohands/frozen_inference_graph.pb";
	else GraphPath = argv[2];

	/*TF_Buffer* buffer;

	if ((buffer = ReadBufferFromFile(GraphPath)) == nullptr) {
		cerr << "Cant read graph at:\t" << GraphPath<<"\n\n";
		return -1;
	}

	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	
	TF_GraphImportGraphDef(graph, buffer, opts, status);
	TF_DeleteImportGraphDefOptions(opts);
	TF_DeleteBuffer(buffer);

	if (TF_GetCode(status) != TF_OK) {
		cerr << "Graph not loaded correctly";
		return -2;
	}
	else cout << "Graph Load success!!\n\n";

	TF_DeleteStatus(status);
	*/
	
	TF_Graph* graph = LoadGraphDef(GraphPath);
	
	if (graph == nullptr) {
		cerr << "Graph not loaded correctly";
		return -1;
	}
	else cout << "Graph Load success!!\n\n";

	string inputLayer = "image_tensor:0";
	vector<string> outputLayer = { "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };


	
	
	
	
	
	
	
	
	return 0;
}