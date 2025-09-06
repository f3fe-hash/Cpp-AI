#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

class ThreadPool
{
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};

public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args);
};

inline ThreadPool::ThreadPool(size_t threads)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this]{
            while (!stop)
            {
                std::function<void()> task;
                {
                    std::unique_lock lock(queue_mutex);
                    condition.wait(lock, [this] {return stop || !tasks.empty();});
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
}

inline ThreadPool::~ThreadPool()
{
    stop = true;
    condition.notify_all();
    for (auto &worker : workers)
        if (worker.joinable())
            worker.join();
}

template<class F, class... Args>
void ThreadPool::enqueue(F&& f, Args&&... args)
{
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
        std::unique_lock lock(queue_mutex);
        tasks.emplace(std::move(task));
    }
    condition.notify_one();
}

// Wait until all tasks are completed
inline void wait_for_tasks(std::atomic<int>& counter, std::condition_variable& cv, std::mutex& mtx, int total)
{
    std::unique_lock lock(mtx);
    cv.wait(lock, [&] { return counter.load() == total; });
}

inline ThreadPool threads{12};
