#pragma once
#include <vstl/compare.h>
#include <vstl/pool.h>
namespace vstd {
namespace detail {
template<typename Node>
class TreeMapUtility {
private:
    static void leftRotate(Node *x, Node *&root) {
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
    static void rightRotate(Node *x, Node *&root) {
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
    static void fixDelete(Node *x, Node *&root, Node *&tNullParent) {
        Node *s;
        bool xIsNull;
        while (x != root && ((xIsNull = (x == nullptr)) || x->color == 0)) {
            auto &&xParent = xIsNull ? tNullParent : x->parent;
            if (x == xParent->left) {
                s = xParent->right;
                if (s->color == 1) {
                    // case 3.1
                    s->color = 0;
                    xParent->color = 1;
                    leftRotate(xParent, root);
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
                        rightRotate(s, root);
                        s = xParent->right;
                    }
                    // case 3.4
                    s->color = xParent->color;
                    xParent->color = 0;
                    s->right->color = 0;
                    leftRotate(xParent, root);
                    x = root;
                    break;
                }
            } else {
                s = xParent->left;
                if (s->color == 1) {
                    // case 3.1
                    s->color = 0;
                    xParent->color = 1;
                    rightRotate(xParent, root);
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
                        leftRotate(s, root);
                        s = xParent->left;
                    }
                    // case 3.4
                    s->color = xParent->color;
                    xParent->color = 0;
                    s->left->color = 0;
                    rightRotate(xParent, root);
                    x = root;
                    break;
                }
            }
        }
        if (x != nullptr)
            x->color = 0;
    }
    static Node *minimum(Node *node) {
        while (node->left != nullptr) {
            node = node->left;
        }
        return node;
    }

    // find the node with the maximum key
    static Node *maximum(Node *node) {
        while (node->right != nullptr) {
            node = node->right;
        }
        return node;
    }

    static void rbTransplant(Node *u, Node *v, Node *&root, Node *&tNullParent) {
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

public:
    static void fixInsert(Node *k, Node *&root) {
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
                        rightRotate(k, root);
                    }
                    // case 3.2.1
                    k->parent->color = 0;
                    k->parent->parent->color = 1;
                    leftRotate(k->parent->parent, root);
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
                        leftRotate(k, root);
                    }
                    // mirror case 3.2.1
                    k->parent->color = 0;
                    k->parent->parent->color = 1;
                    rightRotate(k->parent->parent, root);
                }
            }
            if (k == root) {
                break;
            }
        }
        root->color = 0;
    }
    static void deleteOneNode(Node *z, Node *&root) {
        Node *tNullParent = nullptr;
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
};
}// namespace detail
template<typename K, typename V>
struct TreeElement {
    K first;
    V second;
    template<typename A, typename... B>
    TreeElement(A &&a, B &&...b)
        : first(std::forward<A>(a)),
          second(std::forward<B>(b)...) {
    }
};
template<typename K, typename V>
struct ConstTreeElement {
    const K first;
    V second;
    template<typename A, typename... B>
    ConstTreeElement(A &&a, B &&...b)
        : first(std::forward<A>(a)),
          second(std::forward<B>(b)...) {
    }
};
template<typename K, typename V>
static consteval decltype(auto) TreeElementType() {
    if constexpr (std::is_same_v<V, void>) {
        return TypeOf<K>{};
    } else {
        return TypeOf<TreeElement<K, V>>{};
    }
};
template<typename K, typename V>
static consteval decltype(auto) ConstTreeElementType() {
    if constexpr (std::is_same_v<V, void>) {
        return TypeOf<K>{};
    } else {
        return TypeOf<ConstTreeElement<K, V>>{};
    }
};
}// namespace vstd