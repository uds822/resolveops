

# Error handling with DynamoDB
<a name="Programming.Errors"></a>

 This section describes runtime errors and how to handle them. It also describes error messages and codes that are specific to Amazon DynamoDB. For a list of common errors that apply to all AWS services, see [Access Management](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/CommonErrors.html)

**Topics**
+ [Error components](#Programming.Errors.Components)
+ [Transactional errors](#Programming.Errors.TransactionalErrors)
+ [Error messages and codes](#Programming.Errors.MessagesAndCodes)
+ [Error handling in your application](#Programming.Errors.Handling)
+ [Error retries and exponential backoff](#Programming.Errors.RetryAndBackoff)
+ [Batch operations and error handling](#Programming.Errors.BatchOperations)

## Error components
<a name="Programming.Errors.Components"></a>

When your program sends a request, DynamoDB attempts to process it. If the request is successful, DynamoDB returns an HTTP success status code (`200 OK`), along with the results from the requested operation.

If the request is unsuccessful, DynamoDB returns an error. Each error has three components: 
+ An HTTP status code (such as `400`).
+ An exception name (such as `ResourceNotFoundException`).
+ An error message (such as `Requested resource not found: Table: {{tablename}} not found`).

The AWS SDKs take care of propagating errors to your application so that you can take appropriate action. For example, in a Java program, you can write `try-catch` logic to handle a `ResourceNotFoundException`.

If you are not using an AWS SDK, you need to parse the content of the low-level response from DynamoDB. The following is an example of such a response.

```
HTTP/1.1 400 Bad Request
x-amzn-RequestId: LDM6CJP8RMQ1FHKSC1RBVJFPNVV4KQNSO5AEMF66Q9ASUAAJG
Content-Type: application/x-amz-json-1.0
Content-Length: 240
Date: Thu, 15 Mar 2012 23:56:23 GMT

{"__type":"com.amazonaws.dynamodb.v20120810#ResourceNotFoundException",
"message":"Requested resource not found: Table: {{tablename}} not found"}
```

## Transactional errors
<a name="Programming.Errors.TransactionalErrors"></a>

For information on transactional errors, please see [Transaction conflict handling in DynamoDB](transaction-apis.md#transaction-conflict-handling)

## Error messages and codes
<a name="Programming.Errors.MessagesAndCodes"></a>

The following is a list of exceptions returned by DynamoDB, grouped by HTTP status code. If *OK to retry?* is *Yes*, you can submit the same request again. If *OK to retry?* is *No*, you need to fix the problem on the client side before you submit a new request.

### HTTP status code 400
<a name="Programming.Errors.MessagesAndCodes.http400"></a>

An HTTP `400` status code indicates a problem with your request, such as authentication failure, missing required parameters, or exceeding a table's provisioned throughput. You have to fix the issue in your application before submitting the request again.

**AccessDeniedException **  
Message: *Access denied.*  
The client did not correctly sign the request. If you are using an AWS SDK, requests are signed for you automatically; otherwise, go to the [Signature version 4 signing process](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html) in the *AWS General Reference*.  
OK to retry? No

**ConditionalCheckFailedException**  
Message: *The conditional request failed. *  
You specified a condition that evaluated to false. For example, you might have tried to perform a conditional update on an item, but the actual value of the attribute did not match the expected value in the condition.  
OK to retry? No

**IncompleteSignatureException**  
Message: *The request signature does not conform to AWS standards. *  
The request signature did not include all of the required components. If you are using an AWS SDK, requests are signed for you automatically; otherwise, go to the [Signature Version 4 signing process](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html) in the *AWS General Reference*.  
OK to retry? No

**ItemCollectionSizeLimitExceededException**  
Message: *Collection size exceeded.*  
For a table with a local secondary index, a group of items with the same partition key value has exceeded the maximum size limit of 10 GB. For more information on item collections, see [Item collections in Local Secondary Indexes](LSI.md#LSI.ItemCollections).  
OK to retry? Yes

**LimitExceededException**  
Message: *Too many operations for a given subscriber.*  
There are too many concurrent control plane operations. The cumulative number of tables and indexes in the `CREATING`, `DELETING`, or `UPDATING` state cannot exceed 500.  
OK to retry? Yes

**MissingAuthenticationTokenException**  
Message: *Request must contain a valid (registered) AWS Access Key ID.*  
The request did not include the required authorization header, or it was malformed. See [DynamoDB low-level API](Programming.LowLevelAPI.md).  
OK to retry? No

**ProvisionedThroughputExceededException**  
Message: *You exceeded your maximum allowed provisioned throughput for a table or for one or more global secondary indexes. To view performance metrics for provisioned throughput vs. consumed throughput, open the [Amazon CloudWatch console](https://console.aws.amazon.com/cloudwatch/home).*  
The error includes a list of `ThrottlingReason` fields that provides specific context about why throttling occurred, following the format `ResourceType+OperationType+LimitType` (e.g., `TableReadProvisionedThroughputExceeded`) and the ARN of the affected resource. This helps you identify which resource is being throttled (table or index), what operation type triggered the throttling (read or write), and the specific limit that was exceeded (in this case: provisioned capacity). For more information about diagnosing and resolving throttling issues, see [Diagnosing throttling](throttling-diagnosing-workflow.md).  
Example: Your request rate is too high. The AWS SDKs for DynamoDB automatically retry requests that receive this exception. Your request is eventually successful, unless your retry queue is too large to finish. Reduce the frequency of requests using [Error retries and exponential backoff](#Programming.Errors.RetryAndBackoff).   
OK to retry? Yes

**ReplicatedWriteConflictException**  
Message: *One or more items in this request are being modified by a request in another Region.*  
Example: A write operation was requested for an item in a multi-Region strongly consistent (MRSC) global table that is currently being modified by a request in another Region.   
OK to retry? Yes

**RequestLimitExceeded**  
Message: *Throughput exceeds the current throughput limit for your account. To request a limit increase, contact AWS Support at [https://aws.amazon.com/support](https://aws.amazon.com/support)*.  
The error includes a list of `ThrottlingReason` fields that provides specific context about why throttling occurred, following the format `ResourceType+OperationType+LimitType` (e.g., `TableWriteAccountLimitExceeded` or `IndexReadAccountLimitExceeded`) and the ARN of the affected resource. This helps you identify which resource is being throttled (table or index), what operation type triggered the throttling (read or write), and that you've exceeded account-level service quotas. For more information about diagnosing and resolving throttling issues, see [Diagnosing throttling](throttling-diagnosing-workflow.md).   
Example: Rate of on-demand requests exceeds the allowed account throughput and the table cannot be scaled further.  
OK to retry? Yes

**ResourceInUseException**  
Message: *The resource which you are attempting to change is in use. *  
Example: You tried to re-create an existing table, or delete a table currently in the `CREATING` state.   
OK to retry? No

**ResourceNotFoundException **  
Message: *Requested resource not found.*  
Example: The table that is being requested does not exist, or is too early in the `CREATING` state.  
OK to retry? No

**ThrottlingException**  
Message: *Rate of requests exceeds the allowed throughput.*  
This exception is returned as an AmazonServiceException response with a THROTTLING\_EXCEPTION status code. This exception might be returned if you perform [control plane](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.API.html#HowItWorks.API.ControlPlane) API operations too rapidly.  
For tables using on-demand mode, this exception might be returned for any [data plane](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.API.html#HowItWorks.API.DataPlane) API operation if your request rate is too high. To learn more about on-demand scaling, see [Initial throughput and scaling properties](on-demand-capacity-mode.md#on-demand-capacity-mode-initial).   
The error includes a list of `ThrottlingReason` fields that provides specific context about why throttling occurred, following the format `ResourceType+OperationType+LimitType` (e.g., `TableReadKeyRangeThroughputExceeded` or `IndexWriteMaxOnDemandThroughputExceeded`) and the ARN of the affected resource. This information helps you identify which resource is being throttled (table or index), what operation type triggered the throttling (read or write), and the specific limit that was exceeded (partition limits or on-demand maximum throughput). For more information about diagnosing and resolving throttling issues, see [Diagnosing throttling](throttling-diagnosing-workflow.md).  
OK to retry? Yes

**UnrecognizedClientException**  
Message: *The Access Key ID or security token is invalid.*  
The request signature is incorrect. The most likely cause is an invalid AWS access key ID or secret key.  
OK to retry? Yes

**ValidationException**  
Message: Varies, depending upon the specific error(s) encountered  
This error can occur for several reasons, such as a required parameter that is missing, a value that is out of range, or mismatched data types. The error message contains details about the specific part of the request that caused the error.  
OK to retry? No

### HTTP status code 5xx
<a name="Programming.Errors.MessagesAndCodes.http5xx"></a>

An HTTP `5xx` status code indicates a problem that must be resolved by AWS. This might be a transient error, in which case you can retry your request until it succeeds. Otherwise, go to the [AWS Service Health Dashboard](http://status.aws.amazon.com/) to see if there are any operational issues with the service.

For more information, see [How do I resolve HTTP 5xx errors in Amazon DynamoDB?](https://aws.amazon.com/premiumsupport/knowledge-center/dynamodb-http-5xx-errors/)

**InternalServerError (HTTP 500)**  
DynamoDB could not process your request.  
OK to retry? Yes  
You might encounter internal server errors while working with items. These are expected during the lifetime of a table. Any failed requests can be retried immediately.  
When you receive a status code 500 on a write operation, the operation may have succeeded or failed. If the write operation is a `TransactWriteItem` request, then it is OK to retry the operation. If the write operation is a single-item write request such as `PutItem`, `UpdateItem`, or `DeleteItem`, then your application should read the state of the item before retrying the operation, and/or use [DynamoDB condition expression CLI example](Expressions.ConditionExpressions.md) to ensure that the item remains in a correct state after retrying regardless of whether the prior operation succeeded or failed. If idempotency is a requirement for the write operation, please use [`TransactWriteItem`](transaction-apis.md#transaction-apis-txwriteitems), which supports idempotent requests by automatically specifying a `ClientRequestToken` to disambiguate multiple attempts to perform the same action.

**ServiceUnavailable (HTTP 503)**  
DynamoDB is currently unavailable. (This should be a temporary state.)  
OK to retry? Yes

## Error handling in your application
<a name="Programming.Errors.Handling"></a>

For your application to run smoothly, you need to add logic to catch and respond to errors. Typical approaches include using `try-catch` blocks or `if-then` statements.

The AWS SDKs perform their own retries and error checking. If you encounter an error while using one of the AWS SDKs, the error code and description can help you troubleshoot it.

You should also see a `Request ID` in the response. The `Request ID` can be helpful if you need to work with AWS Support to diagnose an issue.

## Error retries and exponential backoff
<a name="Programming.Errors.RetryAndBackoff"></a>

Numerous components on a network, such as DNS servers, switches, load balancers, and others, can generate errors anywhere in the life of a given request. The usual technique for dealing with these error responses in a networked environment is to implement retries in the client application. This technique increases the reliability of the application.

Each AWS SDK implements retry logic automatically. You can modify the retry parameters to your needs. For example, consider a Java application that requires a fail-fast strategy, with no retries allowed in case of an error. With the AWS SDK for Java, you could use the `ClientConfiguration` class and provide a `maxErrorRetry` value of `0` to turn off the retries. For more information, see the AWS SDK documentation for your programming language.

If you're not using an AWS SDK, you should retry original requests that receive server errors (5xx). However, client errors (4xx, other than a `ThrottlingException` or a `ProvisionedThroughputExceededException`) indicate that you need to revise the request itself to correct the problem before trying again. For detailed recommendations to address specific throttling scenarios, refer to the [DynamoDB throttling troubleshooting](TroubleshootingThrottling.md) section.

In addition to simple retries, each AWS SDK implements an exponential backoff algorithm for better flow control. The concept behind exponential backoff is to use progressively longer waits between retries for consecutive error responses. For example, up to 50 milliseconds before the first retry, up to 100 milliseconds before the second, up to 200 milliseconds before third, and so on. However, after a minute, if the request has not succeeded, the problem might be the request size exceeding your provisioned throughput, and not the request rate. Set the maximum number of retries to stop around one minute. If the request is not successful, investigate your provisioned throughput options. 

**Note**  
The AWS SDKs implement automatic retry logic and exponential backoff.

Most exponential backoff algorithms use jitter (randomized delay) to prevent successive collisions. Because you aren't trying to avoid such collisions in these cases, you do not need to use this random number. However, if you use concurrent clients, jitter can help your requests succeed faster. For more information, see the blog post about [Exponential backoff and jitter](http://www.awsarchitectureblog.com/2015/03/backoff.html).

## Batch operations and error handling
<a name="Programming.Errors.BatchOperations"></a>

The DynamoDB low-level API supports batch operations for reads and writes. `BatchGetItem` reads items from one or more tables, and `BatchWriteItem` puts or deletes items in one or more tables. These batch operations are implemented as wrappers around other non-batch DynamoDB operations. In other words, `BatchGetItem` invokes `GetItem` once for each item in the batch. Similarly,`BatchWriteItem` invokes `DeleteItem` or `PutItem`, as appropriate, for each item in the batch.

A batch operation can tolerate the failure of individual requests in the batch. For example, consider a `BatchGetItem` request to read five items. Even if some of the underlying `GetItem` requests fail, this does not cause the entire `BatchGetItem` operation to fail. However, if all five read operations fail, then the entire `BatchGetItem` fails.

The batch operations return information about individual requests that fail so that you can diagnose the problem and retry the operation. For `BatchGetItem`, the tables and primary keys in question are returned in the `UnprocessedKeys` value of the response. For `BatchWriteItem`, similar information is returned in `UnprocessedItems`. 

The most likely cause of a failed read or a failed write is *throttling*. For `BatchGetItem`, one or more of the tables in the batch request does not have enough provisioned read capacity to support the operation. For `BatchWriteItem`, one or more of the tables does not have enough provisioned write capacity.

If DynamoDB returns any unprocessed items, you should retry the batch operation on those items. However, *we strongly recommend that you use an exponential backoff algorithm*. If you retry the batch operation immediately, the underlying read or write requests can still fail due to throttling on the individual tables. If you delay the batch operation using exponential backoff, the individual requests in the batch are much more likely to succeed.

**Note**  
Some AWS SDKs provide higher-level clients that handle unprocessed item retries automatically, so you don't need to implement this retry logic yourself. For example:  
**Java** – The [DynamoDB Enhanced Client](DynamoDBEnhanced.md) in the AWS SDK for Java v2 and the [DynamoDBMapper](DynamoDBMapper.md) in v1 both automatically retry unprocessed items when performing batch operations.
**Python** – The boto3 Table resource `batch_writer` handles unprocessed item retries implicitly for batch write operations. For more information, see [Using the table resource batch\_writer](programming-with-python.md#programming-with-python-batch-writer).
If you are using a low-level client or an SDK that does not provide this behavior, you must implement the retry logic yourself as described above.