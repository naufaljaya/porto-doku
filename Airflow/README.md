# <img src="https://airflow.apache.org/docs/apache-airflow/1.10.6/_images/pin_large.png" alt="image" width="30" height="auto"> Airflow Project

This section contains two example DAGs for Apache Airflow, each demonstrating different concepts and use cases.

## DAG 1: pool_dag

### Purpose

This DAG fetches forex rates for different currencies using HTTP API requests and displays the results using XCom.

### Components

- Three `SimpleHttpOperator` tasks retrieve forex rates for EUR, USD, and JPY.
- A `BashOperator` task displays task IDs and fetched data from XCom.

### Scheduling

This DAG runs daily at midnight.

### Dependencies

- Parallel execution of currency retrieval tasks.
- `show_data` task runs after currency retrieval tasks.

### Notable Points

- Tasks are assigned to the `'forex_api_pool'` pool for task allocation.
- The template in `bash_command` might require a minor adjustment for XCom usage.

## DAG 2: queue_dag

### Purpose

This DAG demonstrates task queue allocation based on resource requirements (I/O, CPU, Spark dependency).

### Components

- I/O intensive tasks (`t_1_ssd`, `t_2_ssd`, `t_3_ssd`) are assigned to the `'worker_ssd'` queue. - CPU intensive tasks (`t_4_cpu`, `t_5_cpu`) are assigned to the `'worker_cpu'` queue. -
- A task with Spark dependency (`t_6_spark`) is assigned to the `'worker_spark'` queue. - A `DummyOperator` task (`task_7`) acts as a placeholder.

### Scheduling

This DAG runs daily at midnight.

### Dependencies

- I/O and CPU intensive tasks run in parallel.
- `t_6_spark` runs after I/O and CPU tasks.
- All tasks feed into `task_7`.

### Notable Points

- Different queues are used to allocate tasks based on resource requirements.
