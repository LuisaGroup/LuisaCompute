    class Iterator {
    public:
        struct Sentinel {};
    private:
        NodeRef _curr;
        friend class BasicBlock;
        explicit Iterator(NodeRef curr) noexcept
            : _curr{curr} {}
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
        [[nodiscard]] auto operator==(Sentinel) const noexcept { return !_curr.valid(); }
    };
    [[nodiscard]] auto begin() const noexcept { return Iterator{this->first()}; }
    [[nodiscard]] auto end() const noexcept { return Iterator::Sentinel{}; }
    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }