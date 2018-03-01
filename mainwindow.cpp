#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    this->setMouseTracking(true);
    ui->setupUi(this);
    ui->label_3->hide();
    ui->label_3->setDisabled(true);
    ui->actionCost_Graph->setDisabled(true);
    ui->actionPath_Tree->setDisabled(true);
    qApp->installEventFilter(this);


}

QImage Mat2QImage(cv::Mat const& src)
{
    /*
    Mat inv_src(src.cols, src.rows,CV_8UC3, Scalar(255,255,255));
    for(int i = 0; i<inv_src.rows;i++)
        for(int j = 0;j<inv_src.cols;j++)
        {
            inv_src.at<Vec3b>(i,j) = src.at<Vec3b>(j,i);
        }
        */


    cv::Mat temp;
    cvtColor(src, temp,CV_BGR2RGB);
    QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    dest.bits(); // enforce deep copy, see documentation
    // of QImage::QImage ( const uchar * data, int width, int height, Format format )
    return dest;
}

void MainWindow::getMask(cv::Point2d point)
{
    Rect rect;
    floodFill(mask_image,point,Scalar(255,255,255),&rect,Scalar(20,20,20),Scalar(20,20,20));
    show_image(mask_image,1);
    isMask = true;
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if(sum_delta<=2.3&&sum_delta>=-2.3&&event->type() == QEvent::Wheel)
    {
        QWheelEvent *wheelEvent = static_cast<QWheelEvent*>(event);
        float degree = wheelEvent->delta()*1.0/900;

        if(isOpenimage)
        {

                sum_delta+=degree;

            if (sum_delta >=0.1 && current_scale==1)
            {
                    current_scale = 2;
                    Zoom_in_out(current_scale);
                    sum_delta = 0;
            }
            else if(sum_delta >=0.1 && current_scale == 0.5)
            {
                current_scale = 1;
                Zoom_in_out(current_scale);
                sum_delta = 0;
            }
            else if (sum_delta<=-0.1 && current_scale == 2)
            {
                    current_scale =1;
                    Zoom_in_out(current_scale);
                    sum_delta = 0;
            }
            else if (sum_delta<=-0.1 && current_scale == 1)
            {
                current_scale = 0.5;
                Zoom_in_out(current_scale);
                sum_delta = 0;
            }

            cout <<sum_delta<<endl;
        }

    }
    // track mouse move
    if(qobject_cast<QLabel*>(obj)==ui->label &&event->type() == QEvent::MouseMove)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        statusBar()->showMessage(QString("Mouse move (%1,%2)").arg(mouseEvent->pos().x()).arg(mouseEvent->pos().y()));
        if(isOpenimage && isStart){
            Point2d pointb(mouseEvent->pos().x(),mouseEvent->pos().y());
            Mat image_line = contour_image[contour_image.size()-1].clone();
            if ( !points_stack.empty() ) {
                vector<Pixel_Node *>* seed_graph;
                seed_graph = &graphs_stack.top();
                assert(seed_graph != nullptr);
                plot_path_tree_point_to_point(seed_point[seed_point.size()-1],pointb,seed_graph,image_line,false);


            }
 //           cv::line(image_line,seed_point[seed_point.size()-1],pointb,cv::Scalar(255,0,255),2,8,0);
            show_image(image_line,0);
        }

    }
    // mouse click
    if(qobject_cast<QLabel*>(obj)==ui->label &&event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if(mouseEvent->button() == Qt::LeftButton && isStart){
            curr_point.x = mouseEvent->pos().x();
            curr_point.y = mouseEvent->pos().y();
            seed_point.push_back(curr_point);
            Mat image_line = contour_image[contour_image.size()-1].clone();
            //cv::line(image_line,seed_point[seed_point.size()-2],curr_point,cv::Scalar(255,0,255),2,8,0);
            Point mouse_point = Point(curr_point.y,curr_point.x);
            vector<Pixel_Node *> nodes_graph;
            init_node_vector(rows, cols, &nodes_graph, &image_gradient);
            minimum_cost_path_dijkstra(rows, cols, &mouse_point, &nodes_graph);
            if ( !points_stack.empty() ) {

                Point stack = points_stack.top();
                mask_image = mask_vector[mask_vector.size()-1];
//                cout << "clicked: point on stack: " << stack.x << " " << stack.y << " mouse point " << mouse_point.x << " " << mouse_point.y << endl;

                // Draw from the past saved node to the current clicked node
                cv::circle(image_line,curr_point,2,cv::Scalar(255,255,255),2);
                plot_path_tree_point_to_point(seed_point[seed_point.size()-2], curr_point, &graphs_stack.top(), image_line,true);
                        }

                        // Update the stacks
            points_stack.push(mouse_point);
            graphs_stack.push(nodes_graph);
            mask_vector.push_back(mask_image);
            contour_image.push_back(image_line);
            show_image(image_line,0);

        }
        if(mouseEvent->button() == Qt::LeftButton){
            if(QApplication::keyboardModifiers() == Qt::ControlModifier && !isStart){
                isStart = true;
                isClosed = false;
                isMask = false;
                image = image_src.clone();
                sum_delta=0;
                first_point.x = mouseEvent->pos().x();
                first_point.y = mouseEvent->pos().y();
                curr_point = first_point;

                Point start_point = Point(first_point.y, first_point.x);
                vector<Pixel_Node *> nodes_graph;
                init_node_vector(rows, cols, &nodes_graph, &image_gradient);
                minimum_cost_path_dijkstra(rows, cols, &start_point, &nodes_graph);
                points_stack.push(start_point);

                graphs_stack.push(nodes_graph);

                contour_image.clear();
                seed_point.clear();
                mask_image = Mat(image.rows, image.cols,CV_8UC3, Scalar(0,0,0));
                mask_vector.push_back(mask_image);
                seed_point.push_back(curr_point);
                cv::circle(image,first_point,2,cv::Scalar(255,255,255),2);
                contour_image.push_back(image);
                show_image(image,1);
            }
        }
        if(mouseEvent->button() == Qt::RightButton && isStart){
            if(seed_point.size()!=0 && contour_image.size()!=0 && points_stack.size()!=0 && graphs_stack.size()!=0)
            {
                seed_point.pop_back();
                contour_image.pop_back();
                points_stack.pop();
                graphs_stack.pop();
                mask_vector.pop_back();
                if(seed_point.size()==0||contour_image.size()==0)
                {
                    isStart = false;
                    show_image(image,0);
                    seed_point.clear();
                    contour_image.clear();
                }


            }
        }
        if(mouseEvent->button() == Qt::LeftButton){
            if(QApplication::keyboardModifiers() == Qt::ShiftModifier && isClosed){

                curr_point.x = mouseEvent->pos().x();
                curr_point.y = mouseEvent->pos().y();
                getMask(curr_point);
            }
        }


    }

    if(event->type() == QEvent::KeyPress && isStart){
        QKeyEvent *keyEvent = static_cast<QKeyEvent*>(event);
        if((keyEvent->key() == Qt::Key_Enter || keyEvent->key() == Qt::Key_Return)&&!isClosed){
            cout<<"enter click\n";

            show_image(contour_image[contour_image.size()-1],0);
            isStart = false;
        }
        if((seed_point.size()>2)&&(keyEvent->modifiers()==Qt::ControlModifier)&&(keyEvent->key() == Qt::Key_Enter || keyEvent->key() == Qt::Key_Return)&&!isClosed){
            Mat image_line = contour_image[contour_image.size()-1].clone();
            cv::line(image_line,seed_point[seed_point.size()-1],seed_point[0],cv::Scalar(255,0,255),2,8,0);
            show_image(image_line,0);
            contour_image.push_back(image_line);
            isStart = false;
            isClosed = 1;
            mask_image = mask_vector[mask_vector.size()-1];
            cv::line(mask_image,seed_point[seed_point.size()-1],seed_point[0],cv::Scalar(255,255,255),1,8,0);
            show_image(mask_image,1);

        }


    }

    return false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                    this,
                    tr("Open Image"),
                    "/home/pj/qt/text/picture",
    tr("Image files (*.jpg *.png *.tif *.gif);;All files (*.*)") );
    if(filename.length() != 0)
    {
        // some init operation
        ui->label->clear();

        std::string file = filename.toUtf8().constData();
        image = cv::imread(file,1);
        image_src = image.clone();
        rows = image_src.rows;
        cols = image_src.cols;
        coordinate[0] = rows;
        coordinate[1] = cols;
        calculate_cost_image(&image_src, &image_gradient);
        mask_image = Mat(image.rows, image.cols,CV_8UC3, Scalar(0,0,0));
        contour_image.clear();
        seed_point.clear();
        show_image(image,0);
        isOpenimage = true;


    }
}

void MainWindow::show_image(cv::Mat const& src, int n)
{
    //imshow( "image", src );

    QPixmap new_pixmap = QPixmap::fromImage(Mat2QImage( src ));

    int w = new_pixmap.width();
    int h = new_pixmap.height();
    if(n==0)
    {
        ui->label->resize(w,h);
        ui->label->setPixmap(new_pixmap);

    }
    else if (n==1)
    {
        ui->label_2->resize(w,h);
        ui->label_2->setPixmap(new_pixmap);
    }
    else if(n==2)
    {
        ui->label_3->resize(w,h);
        ui->label_3->setPixmap(new_pixmap);
    }

   // ui->scrollArea->show();
}

void MainWindow::on_actionSave_Contour_triggered()
{
    if(!isOpenimage)
        return;
    QString fileName = QFileDialog::getSaveFileName(this,tr("Save File"),"../text/picture/image.jpg","Image files (*.jpg)");
    std::string file = fileName.toUtf8().constData();
    if(!file.empty()){
        imwrite(file,contour_image[contour_image.size()-1]);
    }

}

void MainWindow::on_actionSave_Mask_triggered()
{
    if(!isMask)
        return;

    QString fileName=QFileDialog::getSaveFileName(this,tr("Save File"),"../text/picture/mask_iamge.jpg","Image files (*.jpg)");
    std::string file = fileName.toUtf8().constData();
    if(!file.empty()){
        imwrite( file, mask_image );
    }


}
void MainWindow::Zoom_in_out(float scale)
{
    if(!isOpenimage)
        return;
    Mat temp,scaled_image;
    if(isStart)
        temp = contour_image[contour_image.size()-1].clone();
    else
        temp = image.clone();
    if(scale==1)
    {
        show_image(temp,0);
        return;
    }
    if(scale>1)
        pyrUp(temp,scaled_image,Size(temp.cols * scale,temp.rows *scale));
    else if (scale<1)
        pyrDown(temp,scaled_image,Size(temp.cols * scale,temp.rows *scale));
    show_image(scaled_image,0);

}

int MainWindow::calculate_cost_image(Mat* image_src, Mat* image_gradient)
{
    int rows, cols; // coordinate of the pixel
    rows = image_src->rows;
    cols = image_src->cols;
#ifdef DEBUG
    cout << "rows = " << rows << ", cols = " << cols << endl;
#endif

    // a new picture with nine times the size of original picture, all white pixels
    *image_gradient = Mat((rows - 2) * 3, (cols - 2) * 3, CV_8UC3, Scalar(255, 255, 255));
//    image_gradient->at<Vec3b>( 937,1267 )[0] = 0;

    double D_square[8] = {0};
    int link[8];    // local derivative
    int maxD = 0;   // global maximum derivative
    Vec3b pixel[8];

    int i, j, k, l; // iterators for the original picture
    int x, y;       // iterators for the gradient
    for (i = 1; i < rows - 1; ++i) {
        for (j = 1; j < cols - 1; ++j) {
            // initialize
            for (k = 0; k < 8; ++k) {
                link[k] = 0;
                D_square[k] = 0;
            }

            //// diagonal link,   D(link1)=| img(i+1,j) - img(i,j-1) |/sqrt(2)
            // x + 1, y - 1
            pixel[0] = image_src->at<Vec3b>(i + 1, j);
            pixel[1] = image_src->at<Vec3b>(i, j - 1);

            // x - 1, y - 1
            pixel[2] = image_src->at<Vec3b>(i, j - 1);
            pixel[3] = image_src->at<Vec3b>(i - 1, j);

            // x - 1, y + 1
            pixel[4] = image_src->at<Vec3b>(i - 1, j);
            pixel[5] = image_src->at<Vec3b>(i, j + 1);

            // x + 1, y + 1
            pixel[6] = image_src->at<Vec3b>(i, j + 1);
            pixel[7] = image_src->at<Vec3b>(i + 1, j);

            // Calculate link[1],[3],[5],[7]
            for (k = 0; k < 4; ++k) {
                int m = 2 * k + 1;
                for (l = 0; l < 3; ++l) {
                    D_square[m] += pow(pixel[m - 1][l] - pixel[m][l], 2);
                }
                link[m] = (int) sqrt(D_square[m] / 6);
            }

            //// horizontal link, D(link0)=|(img(i,j-1) + img(i+1,j-1))/2 - (img(i,j+1) + img(i+1,j+1))/2|/2
            // x + 1, y
            pixel[0] = image_src->at<Vec3b>(i, j - 1);
            pixel[1] = image_src->at<Vec3b>(i + 1, j - 1);
            pixel[2] = image_src->at<Vec3b>(i, j + 1);
            pixel[3] = image_src->at<Vec3b>(i + 1, j + 1);

            // x - 1, y
            pixel[4] = image_src->at<Vec3b>(i, j - 1);
            pixel[5] = image_src->at<Vec3b>(i - 1, j - 1);
            pixel[6] = image_src->at<Vec3b>(i, j + 1);
            pixel[7] = image_src->at<Vec3b>(i - 1, j + 1);

            for (l = 0; l < 3; ++l) {
                D_square[0] += pow((pixel[0][l] + pixel[1][l]) / 2 - (pixel[2][l] + pixel[3][l]) / 2, 2);
                D_square[4] += pow((pixel[4][l] + pixel[5][l]) / 2 - (pixel[6][l] + pixel[7][l]) / 2, 2);
            }
            link[0] = (int) sqrt(D_square[0] / 12);
            link[4] = (int) sqrt(D_square[4] / 12);

            //// vertical link,   D(link2)=|(img(i-1,j) + img(i-1,j-1))/2 - (img(i+1,j) + img(i+1,j-1))/2|/2.
            // x    , y - 1
            pixel[0] = image_src->at<Vec3b>(i - 1, j);
            pixel[1] = image_src->at<Vec3b>(i - 1, j - 1);
            pixel[2] = image_src->at<Vec3b>(i + 1, j);
            pixel[3] = image_src->at<Vec3b>(i + 1, j - 1);

            // x    , y + 1
            pixel[4] = image_src->at<Vec3b>(i + 1, j);
            pixel[5] = image_src->at<Vec3b>(i + 1, j + 1);
            pixel[6] = image_src->at<Vec3b>(i - 1, j);
            pixel[7] = image_src->at<Vec3b>(i - 1, j + 1);


            for (l = 0; l < 3; ++l) {
                D_square[2] += pow((pixel[0][l] + pixel[1][l]) / 2 - (pixel[2][l] + pixel[3][l]) / 2, 2);
                D_square[6] += pow((pixel[4][l] + pixel[5][l]) / 2 - (pixel[6][l] + pixel[7][l]) / 2, 2);
            }
            link[2] = (int) sqrt(D_square[2] / 12);
            link[6] = (int) sqrt(D_square[6] / 12);

            //// Find maxD and add the cost graph
            for (k = 0; k < 8; ++k) {
                if (link[l] > maxD)
                    maxD = link[l];
            }

            x = i * 3 - 2;
            y = j * 3 - 2;

            for (k = 0; k < 3; ++k) {
                image_gradient->at<Vec3b>(x, y)[k] = 255;
                image_gradient->at<Vec3b>(x + 1, y - 1)[k] = (uchar) link[0];
                image_gradient->at<Vec3b>(x - 1, y - 1)[k] = (uchar) link[1];
                image_gradient->at<Vec3b>(x - 1, y + 1)[k] = (uchar) link[2];
                image_gradient->at<Vec3b>(x + 1, y + 1)[k] = (uchar) link[3];
                image_gradient->at<Vec3b>(x + 1, y)[k] = (uchar) link[4];
                image_gradient->at<Vec3b>(x - 1, y)[k] = (uchar) link[5];
                image_gradient->at<Vec3b>(x, y - 1)[k] = (uchar) link[6];
                image_gradient->at<Vec3b>(x, y + 1)[k] = (uchar) link[7];
            }
        }
    }


    for (i = 1; i < rows - 1; ++i) {
        for (j = 1; j < cols - 1; ++j) {
            x = i * 3 - 2;
            y = j * 3 - 2;
            //// update cost, cost(link)=(maxD - D(link)) * length(link)
            for (k = 0; k < 3; ++k) {
                image_gradient->at<Vec3b>(x, y)[k] = image_gradient->at<Vec3b>(x, y)[k];
                image_gradient->at<Vec3b>(x + 1, y - 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x + 1, y - 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x - 1, y - 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x - 1, y - 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x - 1, y + 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x - 1, y + 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x + 1, y + 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x + 1, y + 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x + 1, y)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x + 1, y)[k]));
                image_gradient->at<Vec3b>(x - 1, y)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x - 1, y)[k]));
                image_gradient->at<Vec3b>(x, y - 1)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x, y - 1)[k]));
                image_gradient->at<Vec3b>(x, y + 1)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x, y + 1)[k]));
            }

#ifdef DEBUG
//            cout << "x = "  << x << ", y =" << y << endl;
//            cout << "  " << +(uchar)link[0] << ", " << +(uchar)link[1] << endl;
//            cout << "  " << +(uchar)link[2] << ", " << +(uchar)link[3] << endl;
//            cout << "  " << +(uchar)link[4] << ", " << +(uchar)link[5] << endl;
//            cout << "  " << +(uchar)link[6] << ", " << +(uchar)link[7] << endl;
#endif
        }
    }

#ifdef DEBUG
    cout << "maxD = " << maxD << endl;
#endif

    return 0;
}

/**
 * initialize all linked cost for the complete image from image gradient
 * @output node_vector
 * @input image_gradient
 */
void MainWindow::init_node_vector(int rows, int cols, vector<Pixel_Node *> *node_vector, Mat* image_gradient)
{
    node_vector->clear();

    // preallocate the vector to save memory allocation time
    node_vector->reserve((unsigned long) rows * cols);

    int i, j, k, m;
    int x, y;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            auto pixel_node = new Pixel_Node(i, j);

            // Set link cost for normal and edge case
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                for (k = 0; k < 9; ++k)
                    pixel_node->link_cost[k] = INF_COST;
            } else {
                x = i * 3 - 2;
                y = j * 3 - 2;
                int count = 0;
                for (k = -1; k <= 1; ++k) {
                    for (m = -1; m <= 1; ++m) {
                        if (k == 0 && m == 0)
                            pixel_node->link_cost[count] = INF_COST;
                        else
                            pixel_node->link_cost[count] = image_gradient->at<Vec3b>(x + k, y + m)[0];
                        count++;
                    }
                }
            }

            node_vector->push_back(pixel_node);
        }
    }
#ifdef DEBUG_NODE_VECTOR
    int expected_x, expected_y;
    for ( expected_x = 0; expected_x < rows/10; ++expected_x) {
        for ( expected_y = 0; expected_y < cols/10; ++expected_y) {
            auto seed_source = expected_x * cols + expected_y;
            Pixel_Node* current = node_vector[seed_source];
            cout << "expected_x = " << expected_x << " expected_y = " << expected_y << endl;
            current->Print();
        }
    }
#endif
}

/**
 * @brief calculate a minimum cost path for the seed point within a picture
 *          a recursive function from dijkstra's algorithm
 * @input seed, pixel coordinate for a picture
 * @output nodes_graph
 */
bool MainWindow::minimum_cost_path_dijkstra(int rows, int cols, Point *seed, vector<Pixel_Node *> *nodes_graph)
{
    FibHeap active_nodes; // Local priority heap that will be empty in the end

    auto seed_source = seed->x * cols + seed->y;
    Pixel_Node *root = nodes_graph->data()[seed_source];
    root->total_cost = 0;
    active_nodes.Insert(root);

    while (active_nodes.GetNumNodes() > 0) {
        auto current = (Pixel_Node *) active_nodes.ExtractMin();

//        cout << "number of nodes: " << active_nodes.GetNumNodes() << endl;
//        current->Print();

        current->state = Pixel_Node::EXPANDED;

//        if (current->row == dest->x && current->col == dest->y)
//        {
//            // reached destination
//            return true;
//        }

        int i, j;
        int index;
        int x_now, y_now;
        // Expand its neighbor nodes
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                x_now = current->row + i - 1;
                y_now = current->col + j - 1;

                // Keep the index within boundary
                if (x_now >= 0 && x_now < rows && y_now >= 0 && y_now < cols) {
                    index = x_now * cols + y_now;
                    Pixel_Node *neighbor = nodes_graph->data()[index];

//                    neighbor->Print();

                    if (neighbor->state == Pixel_Node::INITIAL) {
                        neighbor->prevNode = current;
                        neighbor->total_cost = current->total_cost + current->link_cost[i * 3 + j];
                        neighbor->state = Pixel_Node::ACTIVE;
                        active_nodes.Insert(neighbor);
                    } else if (neighbor->state == Pixel_Node::ACTIVE) {
                        if (current->total_cost + current->link_cost[i * 3 + j] < neighbor->total_cost) {
                            Pixel_Node new_node(neighbor->row, neighbor->col);
                            new_node = *neighbor; // Get a copy of the original node
                            new_node.total_cost = current->total_cost + current->link_cost[i * 3 + j];
                            new_node.prevNode = current;
                            active_nodes.DecreaseKey(neighbor, new_node);
                        }
                    }
                }
            }
        }
    }
    return true;
}

int MainWindow::plot_path_tree_point_to_point(Point2d seed, Point2d dest, vector<Pixel_Node*> *graph, Mat image_plot,bool drawmask)
{
    int index;
    Pixel_Node *dest_node, *seed_node, *curr_node, *prev_node;
    index = dest.y * image_plot.cols + dest.x;
    dest_node = graph->data()[index];
    assert(dest_node != nullptr);
    cout<<"dest index: "<<index<<endl;

    index = seed.y * image_plot.cols + seed.x;
    cout<<"seed index: "<<index<<endl;
    cout << "col: "<<image_plot.cols<<"  rows: "<<image_plot.rows<<endl;
    seed_node = graph->data()[index];
    assert(seed_node != nullptr);

    curr_node = dest_node;

//    cout << "dest  of the node in plot " << dest_node->row << " col "<< dest_node->col << endl;
//    cout << "start of the node in plot " << curr_node->row << " col "<< curr_node->col << endl;

    // Track back from the graph
    while ( curr_node != nullptr && curr_node->prevNode != nullptr &&
            !(curr_node->row == seed_node->row && curr_node->col == seed_node->col))
    {
        prev_node = curr_node->prevNode;
        // Flip pixels in here too
        auto pointA = Point(curr_node->col, curr_node->row);
        auto pointB = Point(prev_node->col, prev_node->row);
        line(image_plot, pointA, pointB, cv::Scalar(255,0,255),2,8,0);
        if(drawmask)
            line(mask_image, pointA, pointB, cv::Scalar(255,255,255),2,8,0);

        curr_node = prev_node;
    }
//    cout << "end   of the node in plot row " << prev_node->row << " col "<< prev_node->col << endl;

    return 1;
}

void MainWindow::on_actionCost_Graph_triggered()
{
    if(isOpenimage)
    show_image(image_gradient,2);

}

int MainWindow::plot_cost_graph(Mat *image_gradient)
{
    // Create a window
//    namedWindow("gradient window", WINDOW_AUTOSIZE);
//    imshow("gradient window", *image_gradient);

//    vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//    compression_params.push_back(95);
//    try {
//        imwrite(cost_graph_directory, *image_gradient, compression_params);
//    }
//    catch (runtime_error& ex) {
//        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        return 0;
//    }
//    fprintf(stdout, "Saved jpeg file for cost graph.\n");
    return 1;
}

void MainWindow::on_actionWork_Mode_triggered()
{
    ui->label->setEnabled(true);
    ui->label_2->setEnabled(true);
    ui->label->show();
    ui->label_2->show();
//    ui->actionOpen->setEnabled(true);
    ui->actionSave_Contour->setEnabled(true);
    ui->actionSave_Mask->setEnabled(true);

    ui->label_3->hide();
    ui->label_3->setDisabled(true);
    ui->actionCost_Graph->setDisabled(true);
    ui->actionPath_Tree->setDisabled(true);
}




void MainWindow::on_actionDebug_Mode_triggered()
{
    ui->label->setDisabled(true);
    ui->label_2->setDisabled(true);
    ui->label->hide();
    ui->label_2->hide();
//    ui->actionOpen->setDisabled(true);
    ui->actionSave_Contour->setDisabled(true);
    ui->actionSave_Mask->setDisabled(true);

    ui->label_3->setEnabled(true);
    ui->label_3->show();
    ui->actionCost_Graph->setEnabled(true);
    ui->actionPath_Tree->setEnabled(true);

}

void MainWindow::on_actionPath_Tree_triggered()
{
    if(isOpenimage && !graphs_stack.empty())
        plot_path_tree(rows,cols,&graphs_stack.top());
}

int MainWindow::plot_path_tree(int rows, int cols, vector<Pixel_Node*> *graph)
{
    auto path_graph_curr_color = Scalar(255, 191, 0);
    auto path_graph_prev_color = Scalar(127, 50, 0);
    auto complete_path_tree = Mat( rows * 3, cols * 3, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            Pixel_Node* curr = graph->data()[index];
            Pixel_Node* prev = curr->prevNode;
            if (prev != NULL)
            {
                // Draw one point to another
                int x, y;
                x = 3 * i + 1 + prev->col - curr->col;
                y = 3 * j + 1 + prev->row - curr->row;
//                cout << "prev x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_curr_color[0];
                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_curr_color[1];
                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_curr_color[2];
                y = 2 * prev->col + curr->col + 1;
                x = 2 * prev->row + curr->row + 1;
//                cout << "curr x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_prev_color[0];
                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_prev_color[1];
                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_prev_color[2];
            }
        }
    }
    show_image(complete_path_tree,2);
    // Create a window
//    namedWindow("path tree window", WINDOW_AUTOSIZE);
//    imshow("path tree window", complete_path_tree);

//    vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//    compression_params.push_back(95);
//    try {
//        imwrite(path_tree_directory, complete_path_tree, compression_params);
//    }
//    catch (runtime_error& ex) {
//        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        return 0;
//    }
//    fprintf(stdout, "Saved jpeg for path tree.\n");
    return 1;
}

