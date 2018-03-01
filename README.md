# intelligent_scissors
This is my first project of Computer Vision.

##  Environment ##
    -C++ for the core algorithm
    -Qt5 for the UI design
    -Using Ubuntu 16.04, 
     Windows can also support this project. If you want to use windows to run the code, you may need to Install the Qt for 
     Windows and Install the opencv on Windows. Also, you should do some change in the text.pro file to support the windows
     environment.
     
##  Implement ##
    1. Convert the image to a cost graph, where the nodes and the links of the graph are the pixels and the gradient respectivly.
    Each of the pixel connect to eight pixels with links.
    
    2. Computing the minimum path of the cost graph using the Dijkstra's shortest path algorithm.
    
    procedure LiveWireDP

    input: seed, graph

    output: a minimum path tree in the input graph with each node pointing to its predecessor along the minimum cost path to that node from the seed.  Each node will also be assigned a total cost, corresponding to the cost of the the minimum cost path from that node to the seed.

    comment: each node will experience three states: INITIAL, ACTIVE, EXPANDED sequentially. the algorithm terminates when all nodes are EXPANDED. All nodes in graph are initialized as INITIAL. When the algorithm runs, all ACTIVE nodes are kept in a priority queue, pq, ordered by the current total cost from the node to the seed.
    
Begin:

    initialize the priority queue pq to be empty;

    initialize each node to the INITIAL state;

    set the total cost of seed to be zero and make seed the root of the minimum path tree ( pointing to NULL ) ;

    insert seed into pq;

    while pq is not empty 

        extract the node q with the minimum total cost in pq;

        mark q as EXPANDED;

        for each neighbor node r of q  

            if  r has not been EXPANDED

                if  r is still INITIAL

                    make q be the predecessor of r ( for the the minimum path tree );

                    set the total cost of r to be the sum of the total cost of q and link cost from q to r as its total cost;

                    insert r in pq and mark it as ACTIVE;

                else if  r is ACTIVE, e.g., in already in the pq 

                    if the sum of the total cost of q and link cost between q and r is less than the total cost of r

                        update q to be the predecessor of r ( for the minimum path tree );

                        update the total cost of r in pq;

End

## Usage ##
1. Work Mode
When you run the program, the scissors is in the work mode by default.
![work mode](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/UI.png "workmode")

2. Open image
![open](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/open_image.png "")

3.Intelligent cut and get Mask

![get mask](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/cut.png "cut and get mask ")
Ctrl + Mouse Left Click : set the first seed point

Mouse Left Click : set next seed point

Mouse Right Click : delete last seed point

Enter : Finish current contour

Ctrl + Enter : Finish contour as closed

Shift + Mouse Left Click : getMask

Mouse Wheel zoom in / zoom out : zoom in / zoom out

4.Save image with contour and save mask

![save](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/save.png "save")

5.Debug Mode

In this mode you can see the cost graph and path tree.
![debug mode](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/debug_mode.png "")
![cost graph](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/cost_graph.png "cost graph")
![path tree](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/path_tree.png "path tree")

## Artifact ##
![artifact](https://github.com/pjrambo/intelligent_scissors/blob/master/picture/lena_on_chassis.jpg "artifact")






    
