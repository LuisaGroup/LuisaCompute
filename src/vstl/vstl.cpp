
#include <vstl/vstring.h>
#include <vstl/Pool.h>
#include <mutex>
#include <vstl/functional.h>
#include <vstl/Memory.h>
#include <vstl/vector.h>
#include <vstl/MetaLib.h>
//#include "BinaryLinkedAllocator.h"
#include <vstl/TreeMap.h>
void *vengine_default_malloc(size_t sz) {
    return malloc(sz);
}
void vengine_default_free(void *ptr) {
    free(ptr);
}

void *vengine_default_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}
namespace vstd {
namespace detail {

struct Node {
    bool color;  // 1 -> Red, 0 -> Black
    Node *parent;// pointer to the parent
    Node *left;  // pointer to left child
    Node *right; // pointer to right child
};

void leftRotate(void *vx, void *&root) {
    Node *x = reinterpret_cast<Node *>(vx);
    Node *y = x->right;
    x->right = y->left;
    if (y->left != nullptr) {
        y->left->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == nullptr) {
        root = y;
    } else if (x == x->parent->left) {
        x->parent->left = y;
    } else {
        x->parent->right = y;
    }
    y->left = x;
    x->parent = y;
}

// rotate right at node x
void rightRotate(void *vx, void *&root) {
    Node *x = reinterpret_cast<Node *>(vx);
    Node *y = x->left;
    x->left = y->right;
    if (y->right != nullptr) {
        y->right->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == nullptr) {
        root = y;
    } else if (x == x->parent->right) {
        x->parent->right = y;
    } else {
        x->parent->left = y;
    }
    y->right = x;
    x->parent = y;
}
void fixDelete(void *vptr, void *&vRoot, Node *&tNullParent) {
    Node *x = reinterpret_cast<Node *>(vptr);
    Node *root = reinterpret_cast<Node *>(vRoot);
    Node *s;
    bool xIsNull;
    while (x != root && ((xIsNull = (x == nullptr)) || x->color == 0)) {
        auto &&xParent = xIsNull ? *reinterpret_cast<Node **>(&tNullParent) : x->parent;
        if (x == xParent->left) {
            s = xParent->right;
            if (s->color == 1) {
                // case 3.1
                s->color = 0;
                xParent->color = 1;
                leftRotate(xParent, vRoot);
                s = xParent->right;
            }
            bool leftNull = s->left == nullptr || s->left->color == 0;
            bool rightNull = s->right == nullptr || s->right->color == 0;
            if (leftNull && rightNull) {
                // case 3.2
                s->color = 1;
                x = xParent;
            } else {
                if (rightNull) {
                    // case 3.3
                    s->left->color = 0;
                    s->color = 1;
                    rightRotate(s, vRoot);
                    s = xParent->right;
                }

                // case 3.4
                s->color = xParent->color;
                xParent->color = 0;
                s->right->color = 0;
                leftRotate(xParent, vRoot);
                x = root;
            }
        } else {
            s = xParent->left;
            if (s->color == 1) {
                // case 3.1
                s->color = 0;
                xParent->color = 1;
                rightRotate(xParent, vRoot);
                s = xParent->left;
            }
            bool leftNull = s->left == nullptr || s->left->color == 0;
            bool rightNull = s->right == nullptr || s->right->color == 0;
            if (leftNull && rightNull) {
                // case 3.2
                s->color = 1;
                x = xParent;
            } else {
                if (leftNull) {
                    // case 3.3
                    s->right->color = 0;
                    s->color = 1;
                    leftRotate(s, vRoot);
                    s = xParent->left;
                }

                // case 3.4
                s->color = xParent->color;
                xParent->color = 0;
                s->left->color = 0;
                rightRotate(xParent, vRoot);
                x = root;
            }
        }
    }
    if (x != nullptr)
        x->color = 0;
}

void TreeMapUtility::fixInsert(void *vk, void *&vRoot) {
    Node *k = reinterpret_cast<Node *>(vk);
    Node *u;
    while (k->parent->color == 1) {
        if (k->parent == k->parent->parent->right) {
            u = k->parent->parent->left;// uncle
            if (u != nullptr && u->color == 1) {
                // case 3.1
                u->color = 0;
                k->parent->color = 0;
                k->parent->parent->color = 1;
                k = k->parent->parent;
            } else {
                if (k == k->parent->left) {
                    // case 3.2.2
                    k = k->parent;
                    rightRotate(k, vRoot);
                }
                // case 3.2.1
                k->parent->color = 0;
                k->parent->parent->color = 1;
                leftRotate(k->parent->parent, vRoot);
            }
        } else {
            u = k->parent->parent->right;// uncle

            if (u != nullptr && u->color == 1) {
                // mirror case 3.1
                u->color = 0;
                k->parent->color = 0;
                k->parent->parent->color = 1;
                k = k->parent->parent;
            } else {
                if (k == k->parent->right) {
                    // mirror case 3.2.2
                    k = k->parent;
                    leftRotate(k, vRoot);
                }
                // mirror case 3.2.1
                k->parent->color = 0;
                k->parent->parent->color = 1;
                rightRotate(k->parent->parent, vRoot);
            }
        }
        if (k == vRoot) {
            break;
        }
    }
    reinterpret_cast<Node *>(vRoot)->color = 0;
}
Node *minimum(Node *node) {
    while (node->left != nullptr) {
        node = node->left;
    }
    return node;
}

// find the node with the maximum key
Node *maximum(Node *node) {
    while (node->right != nullptr) {
        node = node->right;
    }
    return node;
}

void rbTransplant(Node *u, Node *v, void *&root, Node *&tNullParent) {
    if (u->parent == nullptr) {
        root = v;
    } else if (u == u->parent->left) {
        u->parent->left = v;
    } else {
        u->parent->right = v;
    }
    if (v == nullptr)
        tNullParent = u->parent;
    else
        v->parent = u->parent;
}

void TreeMapUtility::deleteOneNode(void *vz, void *&root) {
    Node *tNullParent = nullptr;
    Node *z = reinterpret_cast<Node *>(vz);
    Node *x;
    Node *y;
    y = z;
    int y_original_color = y->color;
    if (z->left == nullptr) {
        x = z->right;
        rbTransplant(z, z->right, root, tNullParent);
    } else if (z->right == nullptr) {
        x = z->left;
        rbTransplant(z, z->left, root, tNullParent);
    } else {
        y = minimum(z->right);
        y_original_color = y->color;
        x = y->right;
        if (y->parent == z) {
            if (x)
                x->parent = y;
            else
                tNullParent = y;
        } else {
            rbTransplant(y, y->right, root, tNullParent);
            y->right = z->right;
            y->right->parent = y;
        }

        rbTransplant(z, y, root, tNullParent);
        y->left = z->left;
        y->left->parent = y;
        y->color = z->color;
    }
    if (y_original_color == 0) {
        fixDelete(x, root, tNullParent);
    }
}

void *TreeMapUtility::getNext(void *vptr) {
    Node *ptr = reinterpret_cast<Node *>(vptr);
    if (ptr->right == nullptr) {
        Node *pNode;
        while (((pNode = ptr->parent) != nullptr) && (ptr == pNode->right)) {
            ptr = pNode;
        }
        ptr = pNode;
    } else {
        ptr = minimum(ptr->right);
    }
    return ptr;
}
void *TreeMapUtility::getLast(void *vptr) {
    Node *ptr = reinterpret_cast<Node *>(vptr);
    if (ptr->left == nullptr) {
        Node *pNode;
        while (((pNode = ptr->parent) != nullptr) && (ptr == pNode->left)) {
            ptr = pNode;
        }
        if (ptr != nullptr) {
            ptr = pNode;
        }
    } else {
        ptr = maximum(ptr->left);
    }
    return ptr;
}

}// namespace detail
#pragma endregion
}// namespace vstd
#ifdef EXPORT_UNITY_FUNCTION
VENGINE_UNITY_EXTERN void vengine_memcpy(void *dest, void *src, uint64 sz) {
    memcpy(dest, src, sz);
}
VENGINE_UNITY_EXTERN void vengine_memset(void *dest, byte b, uint64 sz) {
    memset(dest, b, sz);
}
VENGINE_UNITY_EXTERN void vengine_memmove(void *dest, void *src, uint64 sz) {
    memmove(dest, src, sz);
}
#endif
