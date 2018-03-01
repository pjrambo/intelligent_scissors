#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QMovie>
#include <QMouseEvent>
#include <QFileDialog>
#include <QWheelEvent>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdio>
#include <iostream>
#include <stack>
#include <chrono>
#include <ctime>
#include "plot.h"
#include "opencv2/highgui/highgui.hpp"

#include "pixel_node.h"
using namespace cv;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    //void moveMouse(int x, int y);
    bool eventFilter(QObject *obj, QEvent *event);
    void on_actionOpen_triggered();
    void show_image(cv::Mat const& src,int n);


    void on_actionSave_Contour_triggered();

    void on_actionSave_Mask_triggered();

//    void mousePressEvent(QMouseEvent *e);
//    void keyPressEvent(QKeyEvent *event);
    void getMask(cv::Point2d point);
    //void slot_wheel_move(QWheelEvent *event);
    void Zoom_in_out(float scale);

    int calculate_cost_image(Mat* image_raw, Mat* image_gradient);
    void init_node_vector(int rows, int cols, vector<Pixel_Node *> *node_vector, Mat* image_gradient);
    bool minimum_cost_path_dijkstra(int rows, int cols, Point *seed, vector<Pixel_Node *> *nodes_graph);
    int plot_path_tree_point_to_point(Point2d seed, Point2d dest, vector<Pixel_Node*> *graph, Mat image_plot,bool drawmask);

    void on_actionCost_Graph_triggered();
    int plot_cost_graph(cv::Mat* image_gradient);

    void on_actionWork_Mode_triggered();

    void on_actionDebug_Mode_triggered();

    void on_actionPath_Tree_triggered();
    int plot_path_tree(int rows, int cols, vector<Pixel_Node*> *graph);

private:
    Ui::MainWindow *ui;
    cv::Mat image;
    cv::Mat image_src;
    cv::Mat image_contour;
    cv::Mat mask_image;
    cv::Point2d first_point;
    cv::Point2d last_point;
    cv::Point2d curr_point;
    std::vector<cv::Mat> contour_image;
    std::vector<cv::Mat> mask_vector;
    std::vector<cv::Point2d> seed_point;


    float current_scale = 1;
    float sum_delta=0;


    bool isOpenimage = false;
    bool isStart = false;
    bool isClosed = false;
    bool isMask = false;

    int first_x;
    int first_y;
    int rows,cols;
    int coordinate[2];
    std::vector<Pixel_Node*> node_vector_original;
    cv::Mat image_gradient, image_path_tree;

    stack< Point > points_stack;
    stack< Mat >   images_stack;
    stack< vector<Pixel_Node*> > graphs_stack;


};

#endif // MAINWINDOW_H
