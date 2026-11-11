

# How Lambda processes records from stream and queue-based event sources
<a name="invocation-eventsourcemapping"></a>

An *event source mapping* is a Lambda resource that reads items from stream and queue-based services and invokes a function with batches of records. Within an event source mapping, resources called *event pollers* actively poll for new messages and invoke functions. By default, Lambda automatically scales event pollers, but for certain event source types, you can use [ provisioned mode](#invocation-eventsourcemapping-provisioned-mode) to control the minimum and maximum number of event pollers dedicated to your event source mapping.

The following services use event source mappings to invoke Lambda Event Functions:
+ [Amazon DocumentDB (with MongoDB compatibility) (Amazon DocumentDB)](with-documentdb.md)
+ [Amazon DynamoDB](with-ddb.md)
+ [Amazon Kinesis](with-kinesis.md)
+ [Amazon MQ](with-mq.md)
+ [Amazon Managed Streaming for Apache Kafka (Amazon MSK)](with-msk.md)
+ [Self-managed Apache Kafka](with-kafka.md)
+ [Amazon Simple Queue Service (Amazon SQS)](with-sqs.md)

**Warning**  
Lambda event source mappings process each event at least once, and duplicate processing of records can occur. To avoid potential issues related to duplicate events, we strongly recommend that you make your function code idempotent. To learn more, see [How do I make my Lambda function idempotent](https://repost.aws/knowledge-center/lambda-function-idempotent) in the AWS Knowledge Center.

## How event source mappings differ from direct triggers
<a name="eventsourcemapping-trigger-difference"></a>

Some AWS services can directly invoke Lambda Event Functions using *triggers*. These services push events to Lambda, and the function is invoked immediately when the specified event occurs. Triggers are suitable for discrete events and real-time processing. When you [create a trigger using the Lambda console](lambda-services.md#lambda-invocation-trigger), the console interacts with the corresponding AWS service to configure the event notification on that service. The trigger is actually stored and managed by the service that generates the events, not by Lambda. Here are some examples of services that use triggers to invoke Lambda Event Functions:
+ **Amazon Simple Storage Service (Amazon S3):** Invokes a function when an object is created, deleted, or modified in a bucket. For more information, see [Tutorial: Using an Amazon S3 trigger to invoke a Lambda Event Function](with-s3-example.md).
+ **Amazon Simple Notification Service (Amazon SNS):** Invokes a function when a message is published to an SNS topic. For more information, see [Tutorial: Using AWS Lambda with Amazon Simple Notification Service](with-sns-example.md).
+ **Amazon API Gateway:** Invokes a function when an API request is made to a specific endpoint. For more information, see [Invoking a Lambda Event Function using an Amazon API Gateway endpoint](services-apigateway.md).

Event source mappings are Lambda resources created and managed within the Lambda service. Event source mappings are designed for processing high-volume streaming data or messages from queues. Processing records from a stream or queue in batches is more efficient than processing records individually. 

## Batching behavior
<a name="invocation-eventsourcemapping-batching"></a>

By default, an event source mapping batches records together into a single payload that Lambda sends to your Event Function. To fine-tune batching behavior, you can configure a batching window ([MaximumBatchingWindowInSeconds](https://docs.aws.amazon.com/lambda/latest/api/API_CreateEventSourceMapping.html#lambda-CreateEventSourceMapping-request-MaximumBatchingWindowInSeconds)) and a batch size ([BatchSize](https://docs.aws.amazon.com/lambda/latest/api/API_CreateEventSourceMapping.html#lambda-CreateEventSourceMapping-response-BatchSize)). A batching window is the maximum amount of time to gather records into a single payload. A batch size is the maximum number of records in a single batch. Lambda invokes your function when one of the following three criteria is met:
+ **The batching window reaches its maximum value.** Default batching window behavior varies depending on the specific event source.
  + **For Kinesis, DynamoDB, and Amazon SQS event sources:** The default batching window is 0 seconds. This means that Lambda invokes your function as soon as records are available. To set a batching window, configure `MaximumBatchingWindowInSeconds`. You can set this parameter to any value from 0 to 300 seconds in increments of 1 second. If you configure a batching window, the next window begins as soon as the previous function invocation completes.
  + **For Amazon MSK, self-managed Apache Kafka, Amazon MQ, and Amazon DocumentDB event sources:** The default batching window is 500 ms. You can configure `MaximumBatchingWindowInSeconds` to any value from 0 seconds to 300 seconds in increments of seconds. In provisioned mode for Kafka event source mappings, when you configure a batching window, the next window begins as soon as the previous batch is completed. In non-provisioned Kafka event source mappings, when you configure a batching window, the next window begins as soon as the previous function invocation completes.

    To minimize latency when using Kafka event source mappings in provisioned mode, set `MaximumBatchingWindowInSeconds` to 0. This setting ensures that Lambda starts processing the next batch immediately after completing the current function invocation. For additional information on low latency processing, see [Low latency Apache Kafka](with-kafka-low-latency.md).
  + **For Amazon MQ and Amazon DocumentDB event sources:** The default batching window is 500 ms. You can configure `MaximumBatchingWindowInSeconds` to any value from 0 seconds to 300 seconds in increments of seconds. A batching window begins as soon as the first record arrives.
**Note**  
Because you can only change `MaximumBatchingWindowInSeconds` in increments of seconds, you cannot revert to the 500 ms default batching window after you have changed it. To restore the default batching window, you must create a new event source mapping.
+ **The batch size is met.** The minimum batch size is 1. The default and maximum batch size depend on the event source. For details about these values, see the [BatchSize](https://docs.aws.amazon.com/lambda/latest/api/API_CreateEventSourceMapping.html#lambda-CreateEventSourceMapping-request-BatchSize) specification for the `CreateEventSourceMapping` API operation.
+ **The payload size reaches [6 MB](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html).** You cannot modify this limit.

The following diagram illustrates these three conditions. Suppose a batching window begins at `t = 7` seconds. In the first scenario, the batching window reaches its 40 second maximum at `t = 47` seconds after accumulating 5 records. In the second scenario, the batch size reaches 10 before the batching window expires, so the batching window ends early. In the third scenario, the maximum payload size is reached before the batching window expires, so the batching window ends early.

![Batching window expires when max time reached, batch size met, or payload hits 6 MB.](https://docs.aws.amazon.com/lambda/latest/dg/images/batching-window.png)


We recommend that you test with different batch and record sizes so that the polling frequency of each event source is tuned to how quickly your Event Function is able to complete its task. The [CreateEventSourceMapping](https://docs.aws.amazon.com/lambda/latest/api/API_CreateEventSourceMapping.html) `BatchSize` parameter controls the maximum number of records that can be sent to your function with each invoke. A larger batch size can often more efficiently absorb the invoke overhead across a larger set of records, increasing your throughput.

Lambda doesn't wait for any configured [extensions](lambda-extensions.md) to complete before sending the next batch for processing. In other words, your extensions may continue to run as Lambda processes the next batch of records. This can cause throttling issues if you breach any of your account's [concurrency](lambda-concurrency.md) settings or limits. To detect whether this is a potential issue, monitor your functions and check whether you're seeing higher [concurrency metrics](monitoring-concurrency.md#general-concurrency-metrics) than expected for your event source mapping. Due to short times in between invokes, Lambda may briefly report higher concurrency usage than the number of shards. This can be true even for Lambda Event Functions without extensions.

By default, if your function returns an error, the event source mapping reprocesses the entire batch until the function succeeds, or the items in the batch expire. To ensure in-order processing, the event source mapping pauses processing for the affected shard until the error is resolved. For stream sources (DynamoDB and Kinesis), you can configure the maximum number of times that Lambda retries when your function returns an error. Service errors or throttles where the batch does not reach your function do not count toward retry attempts. You can also configure the event source mapping to send an invocation record to a [destination](invocation-async-retain-records.md#invocation-async-destinations) when it discards an event batch.

## Provisioned mode
<a name="invocation-eventsourcemapping-provisioned-mode"></a>

Lambda event source mappings use event pollers to poll your event source for new messages. By default, Lambda manages the autoscaling of these pollers based on message volume. When message traffic increases, Lambda automatically increases the number of event pollers to handle the load, and reduces them when traffic decreases.

In provisioned mode, you can fine-tune the throughput of your event source mapping by defining minimum and maximum limits for dedicated polling resources that remain ready to handle expected traffic patterns. These resources auto-scale 3 times faster to handle sudden spikes in event traffic and provide 16 times higher capacity to process millions of events. This helps you build highly responsive event-driven workloads with stringent performance requirements.

In Lambda, an event poller is a compute unit with throughput capabilities that vary by event source type. For Amazon MSK and self-managed Apache Kafka, each event poller can handle up to 5 MB/sec of throughput or up to 5 concurrent invocations. For example, if your event source produces an average payload of 1 MB and the average duration of your function is 1 second, a single Kafka event poller can support 5 MB/sec throughput and 5 concurrent Lambda invocations (assuming no payload transformation). For Amazon SQS, each event poller can handle up to 1 MB/sec of throughput or up to 10 concurrent invocations. Using provisioned mode incurs additional costs based on your event poller usage. For pricing details, see [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/).

Provisioned mode is available for Amazon MSK, self-managed Apache Kafka, and Amazon SQS event sources. While concurrency settings give you control over the scaling of your function, provisioned mode gives you control over the throughput of your event source mapping. To ensure maximum performance, you might need to adjust both settings independently.

Provisioned mode is ideal for real-time applications requiring consistent event processing latency, such as financial services firms processing market data feeds, e-commerce platforms providing real-time personalized recommendations, and gaming companies managing live player interactions.

Each event poller supports different throughput capacity:
+ For Amazon MSK and self-managed Apache Kafka: up to 5 MB/sec of throughput or up to 5 concurrent invokes
+ For Amazon SQS: up to 1 MB/sec of throughput or up to 10 concurrent invokes or up to 10 SQS polling API calls per second.

For Amazon SQS event source mappings, you can set the minimum number of pollers between 2 and 200 with a default of 2, and the maximum number between 2 and 10,000 with a default of 200. Lambda scales the number of event pollers between your configured minimum and maximum, quickly adding up to 1,000 concurrency per minute to provide consistent, low-latency processing of your events.

For Kafka event source mappings, you can set the minimum number of pollers between 1 and 200 with a default of 1, and the maximum number between 1 and 2,000 with a default of 200. Lambda scales the number of event pollers between your configured minimum and maximum based on your event backlog in your topic to provide low-latency processing of your events.

Note that for Amazon SQS event sources, the maximum concurrency setting cannot be used with provisioned mode. When using provisioned mode, you control concurrency through the maximum event pollers setting.

For details about configuring provisioned mode, see the following sections:
+ [Configuring provisioned mode for Amazon MSK event source mappings](kafka-scaling-modes.md)
+ [Configuring provisioned mode for self-managed Apache Kafka event source mappings](kafka-scaling-modes.md#kafka-provisioned-mode)
+ [Using provisioned mode with Amazon SQS event source mappings](with-sqs.md#sqs-provisioned-mode)

To minimize latency in provisioned mode, set `MaximumBatchingWindowInSeconds` to 0. This setting ensures that Lambda starts processing the next batch immediately after completing the current function invocation. For additional information on low latency processing, see [Low latency Apache Kafka](with-kafka-low-latency.md).

After configuring provisioned mode, you can observe the usage of event pollers for your workload by monitoring the `ProvisionedPollers` metric. For more information, see [Event source mapping metrics](monitoring-metrics-types.md#event-source-mapping-metrics).

## Event source mapping API
<a name="event-source-mapping-api"></a>

To manage an event source with the [AWS Command Line Interface (AWS CLI)](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) or an [AWS SDK](https://aws.amazon.com/getting-started/tools-sdks/), you can use the following API operations:
+ [CreateEventSourceMapping](https://docs.aws.amazon.com/lambda/latest/api/API_CreateEventSourceMapping.html)
+ [ListEventSourceMappings](https://docs.aws.amazon.com/lambda/latest/api/API_ListEventSourceMappings.html)
+ [GetEventSourceMapping](https://docs.aws.amazon.com/lambda/latest/api/API_GetEventSourceMapping.html)
+ [UpdateEventSourceMapping](https://docs.aws.amazon.com/lambda/latest/api/API_UpdateEventSourceMapping.html)
+ [DeleteEventSourceMapping](https://docs.aws.amazon.com/lambda/latest/api/API_DeleteEventSourceMapping.html)