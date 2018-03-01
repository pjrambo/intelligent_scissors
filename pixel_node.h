#ifndef INTELLIGENT_SCISSOR_PIXEL_NODE_H
#define INTELLIGENT_SCISSOR_PIXEL_NODE_H

#include <cstdio>
#include <iostream>
#include "fibheap.h"

#define INF_COST 0x0FFFFFFF

using namespace std;

/**
 * Pixel node structure
 * order of the cost changed, different from the counter clockwise in image gradient
 * upper left link as link_cost[0], increment left to right
 * link cost to itself as link_cost[4] is zero
 */
class Pixel_Node : public FibHeapNode
{
public:
    int state;
    int row, col;

    int link_cost[9];
    long total_cost;

    Pixel_Node* prevNode; // connecting to multiple other nodes called graph

    enum Node_state{INITIAL, ACTIVE, EXPANDED};
    // constructor
    Pixel_Node(int row, int col) : FibHeapNode()
    {
        this->state        = INITIAL;
        this->row          = row;
        this->col          = col;
        this->total_cost   = INF_COST;
        this->prevNode     = nullptr;
        for (int i = 0; i < 9; ++i) { this->link_cost[i] = INF_COST; }
    }

    virtual void operator =  (FibHeapNode &RHS);
    virtual int  operator == (FibHeapNode &RHS);
    virtual int  operator <  (FibHeapNode &RHS);

    virtual void Print();
};


#endif //INTELLIGENT_SCISSOR_PIXEL_NODE_H
