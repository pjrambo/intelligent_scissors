#ifndef INTELLIGENT_SCISSOR_PLOT_H
#define INTELLIGENT_SCISSOR_PLOT_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include "pixel_node.h"

using namespace cv;

//int plot_cost_graph(cv::Mat* image_gradient);

int plot_path_tree(int rows, int cols, vector<Pixel_Node*> *graph);

//int plot_path_tree_point_to_point(cv::Point* seed, cv::Point* dest, vector<Pixel_Node*> *graph, cv::Mat* image_plot);

#endif //INTELLIGENT_SCISSOR_PLOT_H
