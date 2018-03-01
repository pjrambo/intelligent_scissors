#ifndef INTELLIGENT_SCISSOR_H
#define INTELLIGENT_SCISSOR_H

#include "pixel_node.h"
using namespace cv;

int calculate_cost_image(Mat* image_src, Mat* image_gradient);
void init_node_vector(int rows, int cols, vector<Pixel_Node *> *node_vector, Mat* image_gradient);
bool minimum_cost_path_dijkstra(int rows, int cols, Point *seed, vector<Pixel_Node *> *nodes_graph);

// Global node graph
std::vector<Pixel_Node*> node_vector_original;
cv::Mat image_src, image_gradient, image_path_tree;

#endif //INTELLIGENT_SCISSOR_H
