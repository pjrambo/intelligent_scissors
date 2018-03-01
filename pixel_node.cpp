#include "pixel_node.h"

void Pixel_Node::Print()
{
    FibHeapNode::Print();
    cout << "state: " << state << " row: " << row << " col: " << col << " cost: " << total_cost << endl;
}

void Pixel_Node::operator = (FibHeapNode &RHS)
{
    auto pRHS = (Pixel_Node&) RHS;
    FHN_Assign(RHS);
    this->state  = pRHS.state;
    this->row    = pRHS.row;
    this->col    = pRHS.col;
    this->total_cost = pRHS.total_cost;
    this->prevNode   = pRHS.prevNode;
    for (int i = 0; i < 9; ++i) { this->link_cost[i] = pRHS.link_cost[i]; }
}

int Pixel_Node::operator == (FibHeapNode &RHS)
{
    auto pRHS = (Pixel_Node&) RHS;

    // Make sure both sides are not negative infinite
    if (FHN_Cmp(RHS)) return 0;

//    return (this->row == pRHS.row && this->col == pRHS.col) ? 1 : 0;

    // Misunderstand the ==, should be comparing the cost
    return total_cost == pRHS.total_cost;
}


int Pixel_Node::operator < (FibHeapNode &RHS)
{
    int X;
    if ((X = FHN_Cmp(RHS)) != 0)
        return X < 0 ? 1 : 0;

    // For priority queue "push with priority", if the cost is high, the priority is low
    // For fibonacci node, the priority is high when the return number is large
    return this->total_cost < ((Pixel_Node&) RHS).total_cost ? 1 : 0;
}
