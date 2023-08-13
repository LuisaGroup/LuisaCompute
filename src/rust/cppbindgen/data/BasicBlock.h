    class Iterator {
    public:
        struct Sentinel {};
    private:
        NodeRef _curr;
        NodeRef _end;
        friend class BasicBlock;
        Iterator(NodeRef curr, NodeRef end) noexcept
            : _curr{curr}, _end{end} {}
    public:
        [[nodiscard]] auto operator*() const noexcept { return _curr; }
        auto &operator++() noexcept {
            auto node = _curr->next();
            _curr = node;
            return *this;
        }
        auto operator++(int) noexcept {
            auto old = *this;
            ++(*this);
            return old;
        }
        [[nodiscard]] auto operator==(const Iterator &rhs) const noexcept { return _curr == rhs._curr; }
    };
    [[nodiscard]] auto begin() const noexcept { return Iterator{this->first()->next(), this->last()}; }
    [[nodiscard]] auto end() const noexcept { return Iterator{this->last(), this->last()}; }
    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }