#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>

namespace tokenizer {

class PersistentThreadPool {
public:
    static PersistentThreadPool& instance() {
        static PersistentThreadPool pool;
        return pool;
    }

    [[nodiscard]] size_t worker_count() const {
        return workers_.size();
    }

    void parallel_for (size_t total_items, const std::function<void(size_t index)>& task) {
        if (total_items == 0) return;
        if (total_items == 1 || workers_.empty()) {
            for (size_t i = 0; i < total_items; ++i) {
                task(i);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            current_task_ = &task;
            total_items_.store(total_items, std::memory_order_relaxed);
            next_item_.store(0, std::memory_order_relaxed);
            remaining_items_.store(total_items, std::memory_order_release);
            ++generation_; 
        }

        cv_work_.notify_all();

        while (true) {
            size_t idx = next_item_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_items) break;
            task(idx);
            if (remaining_items_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::unique_lock<std::mutex> lock(done_mutex_);
                cv_done_.notify_all();
            }
        }

        std::unique_lock<std::mutex> lock(done_mutex_);
        cv_done_.wait(lock, [this] () {
            return remaining_items_.load(std::memory_order_acquire) == 0;
        });

        current_task_ = nullptr;
    }

    ~PersistentThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
            ++generation_;
        }
        cv_work_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
private:
    PersistentThreadPool() {
        const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());

        const size_t num_workers = hw > 1 ? hw - 1 : 1;
        workers_.reserve(num_workers);

        for (size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this]() {
                this->worker_loop();
            });
        }
    }

    void worker_loop() {
        uint64_t last_gen = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_work_.wait(lock, [this, last_gen]() {
                return stop_ || generation_ > last_gen;
            });
            if (stop_) break;
            last_gen = generation_;
            lock.unlock();

            const size_t total = total_items_.load(std::memory_order_relaxed);
            while (true) {
                size_t idx = next_item_.fetch_add(1, std::memory_order_relaxed);
                if (idx >= total) break;

                (*current_task_)(idx);

                if (remaining_items_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> done_lock(done_mutex_);
                    cv_done_.notify_one();
                }
            }
        }
    }
    std::vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable cv_work_;
    std::mutex done_mutex_;
    std::condition_variable cv_done_;

    const std::function<void(size_t index)>* current_task_{nullptr};
    std::atomic<size_t> total_items_{0};
    std::atomic<size_t> next_item_{0};
    std::atomic<size_t> remaining_items_{0};
    uint64_t generation_{0};
    bool stop_{false};
};

}