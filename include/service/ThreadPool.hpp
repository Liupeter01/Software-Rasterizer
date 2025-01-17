#pragma once
#ifndef _THREADPOOL_HPP
#define _THREADPOOL_HPP
#include <atomic>
#include <condition_variable>
#include <future>
#include <queue>
#include <service/Noncopyable.hpp>
#include <service/Singleton.hpp>
#include <thread>
#include <vector>

namespace SoftRasterizer {
/*forward declaration*/

class ThreadPool : public Singleton<ThreadPool>, public Noncopyable {

  using Task = std::packaged_task<void()>;

public:
  ThreadPool(std::size_t number = std::thread::hardware_concurrency());
  virtual ~ThreadPool();

public:
  void terminate(); // Terminate ThreadPool

  /*Commit Task To Queue*/
  template <typename F, typename... Args>
  std::future<typename std::invoke_result_t<F, Args...>> /*Ret Value*/
  commit(F &&f, Args... args) {
    using RetType = typename std::invoke_result_t<F, Args...>;
    using FuncType = std::packaged_task<RetType(void)>;

    /*If Thread Pool Closed!*/
    if (m_stopFlag) {
      throw std::runtime_error("Thread pool is stopped.");
    }

    auto task = std::make_shared<FuncType>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<RetType> retValue = task->get_future();

    {
      std::unique_lock<std::mutex> _lckg(m_mtx);
      m_tasks.push(std::make_unique<Task>([task]() -> void { (*task)(); }));
    }

    // Because we insert One Task, So just notify one thead to execute!
    this->m_cv.notify_one();

    return retValue;
  }

protected:
  void initialize(); // Start ThreadPool
  void
  worker(std::size_t thread_id); // Working Thread, Used For Create Thread Pool

private:
  std::mutex m_mtx;
  std::condition_variable m_cv;

  /*How Many Threads That Are Available!*/
  std::atomic<std::size_t> m_threads;

  std::atomic<bool> m_stopFlag;              // indicate pool is still running
  std::vector<std::thread> m_pool;           // Thread Pool
  std::queue<std::unique_ptr<Task>> m_tasks; // Task List
};
} // namespace SoftRasterizer

#endif //_THREADPOOL_HPP
