#include <memory>
#include <service/ThreadPool.hpp>

SoftRasterizer::ThreadPool::ThreadPool(std::size_t number)
          : m_stopFlag(false)
          , m_threads(number <= 1 ? 2 : number)
{
          try{
                    //Start Thread Pool
                    initialize();
          }
          catch (const std::exception& e){
                    terminate();
          }
}

SoftRasterizer::ThreadPool::~ThreadPool() {
          teminate();
}

/*Start ThreadPool*/
void
SoftRasterizer::ThreadPool::initialize() {
          /*resize pool according to atomic variable*/
          m_pool.resize(m_threads);

          for (std::size_t i = 0; i < m_threads; ++i) {
                    m_pool.emplace_back(&SoftRasterizer::ThreadPool::worker, this, i);
          }
}

/*Terminate Thread Pool*/
void 
SoftRasterizer::ThreadPool::teminate() {
          /*Stop Flag = True*/
          m_stopFlag.store(true);

          /*Notify All Suspened Thread To Wake up and terminate*/
          m_cv.notify_all();

          for (auto& thread : m_pool) {
                    if (thread.joinable()) {
                              thread.join();
                    }
          }
}

void 
SoftRasterizer::ThreadPool::worker(std::size_t thread_id) {
          while (true){
                    std::unique_lock<std::mutex> _lckg(m_mtx);
                    m_cv.wait(_lckg, [this]() {return !m_tasks.empty() || m_stopFlag; });

                    //Avaiable Thread Number Minus one
                    --this->m_threads;

                    if (m_stopFlag) {

                              /*ThreadPool Terminated! Processing All Rest Data Then ShutDown*/
                              while (!m_tasks.empty()) {
                                        auto task = std::move(m_tasks.front());
                                        (*task)();
                                        m_tasks.pop();
                              }
                              break;
                    }

                    auto task = std::move(m_tasks.front());  //Get Task From Queue
                    m_tasks.pop();

                    (*task)();

                    //Current Thread Remain Avaiable!
                    ++this->m_threads;
          }
}