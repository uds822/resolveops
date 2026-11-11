

# Locking objects with Object Lock
<a name="object-lock"></a>

S3 Object Lock can help prevent Amazon S3 objects from being deleted or overwritten for a fixed or variable amount of time, or indefinitely. Object Lock uses a *write-once-read-many* (WORM) model to store objects. You can use Object Lock to help meet regulatory requirements that require WORM storage, or to add another layer of protection against object changes or deletion.

**Note**  
S3 Object Lock has been assessed by Cohasset Associates for use in environments that are subject to SEC 17a-4, CFTC, and FINRA regulations. For more information about how Object Lock relates to these regulations, see the [Cohasset Associates Compliance Assessment](https://d1.awsstatic.com/r2018/b/S3-Object-Lock/Amazon-S3-Compliance-Assessment.pdf).

Object Lock provides two ways to manage object retention: *retention periods* and *legal holds*. An object version can have a retention period, a legal hold, or both.
+ **Retention period** – A retention period specifies a fixed period of time during which an object version remains locked. S3 Object Lock provides two retention types: fixed and variable. With fixed retention, you specify the retain-until-date. With variable retention, you specify an event hold and an event hold duration, and Amazon S3 sets the retain-until-date when you release the hold. You can set a unique retention period for individual objects. Additionally, you can set a default retention period on an S3 bucket. You may also restrict the minimum and maximum allowable retention periods with the `s3:object-lock-remaining-retention-days` condition key in the bucket policy. This condition key helps you establish the allowable retention period. For more information, see [Setting limits on retention periods with a bucket policy](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-managing.html#object-lock-managing-retention-limits).
+ **Legal hold** – A legal hold provides the same protection as a retention period, but it has no expiration date. Instead, a legal hold remains in place until you explicitly remove it. Legal holds are independent from retention periods and are placed on individual object versions.

Object Lock works only in buckets that have S3 Versioning enabled. When you lock an object version, Amazon S3 stores the lock information in the metadata for that object version. Placing a retention period or a legal hold on an object protects only the version that's specified in the request. Retention periods and legal holds don't prevent new versions of the object from being created, or delete markers to be added on top of the object. For information about S3 Versioning, see [Retaining multiple versions of objects with S3 Versioning](Versioning.md).

If you put an object into a bucket that already contains an existing protected object with the same object key name, Amazon S3 creates a new version of that object. The existing protected version of the object remains locked according to its retention configuration.

**Note**  
Objects protected by Object Lock (both GOVERNANCE and COMPLIANCE modes) do not allow annotation modifications (create, update, or delete). The `BypassGovernanceRetention` permission does not apply to annotation operations. To add annotations to a locked object, create a new object version. The new version is not subject to the previous version's retention settings unless you explicitly apply retention to it.

## How S3 Object Lock works
<a name="object-lock-overview"></a>

**Topics**
+ [Retention periods](#object-lock-retention-periods)
+ [Retention modes](#object-lock-retention-modes)
+ [Retain-until-date behavior with variable retention](#object-lock-variable-retention-date-behavior)
+ [Legal holds](#object-lock-legal-holds)
+ [How deletes work with S3 Object Lock](#object-lock-how-deletes-work)
+ [Best practices for using S3 Object Lock](#object-lock-best-practices)
+ [Required permissions](#object-lock-permissions)

### Retention periods
<a name="object-lock-retention-periods"></a>

A *retention period* protects an object version for a fixed or variable amount of time. Each retention period has a retain-until-date that indicates when the protection expires. After the retention period expires, the object version can be overwritten or deleted.

You can place a retention period explicitly on an individual object version or on a bucket's properties so that it applies to all objects in the bucket automatically. When you apply a fixed retention period to an object version explicitly, you specify a *Retain Until Date* for the object version, and Amazon S3 stores that date in the object version's metadata.

Object Lock provides two types of retention periods: *fixed* and *variable*. With fixed retention, you specify the retain-until-date when you apply the retention period. With *variable retention*, you defer the retain-until-date. You turn on an *event hold* and specify an *event hold duration*, and the object stays protected while the event hold is on. When you release the hold, Amazon S3 sets the retain-until-date to the release time plus the duration.

You can also specify a retain-until-date with variable retention. Amazon S3 treats that date as a minimum, so the object stays protected until at least that date, even if you release the event hold earlier.

You can also set a retention period in a bucket's properties. When you set a retention period on a bucket, you specify a duration, in either days or years, for how long to protect every object version placed in the bucket. When you place an object in the bucket, Amazon S3 calculates a *Retain Until Date* for the object version by adding the specified duration to the object version's creation timestamp. The object version is then protected exactly as though you explicitly placed an individual lock with that retention period on the object version. A bucket's default retention can also turn on an event hold, so that every object version placed in the bucket starts with variable retention instead of a fixed retain-until-date.

**Note**  
When you `PUT` an object version that has an explicit individual retention mode and period in a bucket, the object version's individual Object Lock settings override any bucket property retention settings.

Like all other Object Lock settings, retention periods apply to individual object versions. Different versions of a single object can have different retention modes and periods.

For example, suppose that you have an object that is 15 days into a 30-day retention period, and you `PUT` an object into Amazon S3 with the same name and a 60-day retention period. In this case, your `PUT` request succeeds, and Amazon S3 creates a new version of the object with a 60-day retention period. The older version maintains its original retention period and becomes deletable in 15 days.

After you've applied a retention setting to an object version, you can extend the retention period. To do this, submit a new Object Lock request for the object version with a *Retain Until Date* that is later than the one currently configured for the object version. Amazon S3 replaces the existing retention period with the new, longer period. Any user with permissions to place an object retention period can extend a retention period for an object version. To set a retention period, you must have the `s3:PutObjectRetention` permission.

While an event hold is on, you can also change the event hold duration. Amazon S3 does not set the retain-until-date to an earlier date than its current value.

When you set a retention period on an object or S3 bucket, you must select one of two retention modes: *compliance* or *governance*.

### Retention modes
<a name="object-lock-retention-modes"></a>

S3 Object Lock provides two retention modes that apply different levels of protection to your objects:
+ Compliance mode
+ Governance mode

In *compliance* mode, a protected object version can't be overwritten or deleted by any user, including the root user in your AWS account. When an object is locked in compliance mode, its retention mode can't be changed, and its retention period can't be shortened. Compliance mode helps ensure that an object version can't be overwritten or deleted for the duration of the retention period.

**Note**  
The only way to delete an object under the compliance mode before its retention date expires is to delete the associated AWS account.

In *governance* mode, users can't overwrite or delete an object version or alter its lock settings unless they have special permissions. With governance mode, you protect objects against being deleted by most users, but you can still grant some users permission to alter the retention settings or delete the objects if necessary. You can also use governance mode to test retention-period settings before creating a compliance-mode retention period. 

To override or remove governance-mode retention settings, you must have the `s3:BypassGovernanceRetention` permission and must explicitly include `x-amz-bypass-governance-retention:true` as a request header with any request that requires overriding governance mode. 

**Note**  
By default, the Amazon S3 console includes the `x-amz-bypass-governance-retention:true` header. If you try to delete objects protected by *governance* mode and have the `s3:BypassGovernanceRetention` permission, the operation will succeed. 

### Retain-until-date behavior with variable retention
<a name="object-lock-variable-retention-date-behavior"></a>

The retain-until-date tells you when an object can become deletable. With fixed retention, this date doesn't change. With variable retention, Amazon S3 computes the date while the event hold is on, and fixes it when you release the hold. The following examples describe this behavior in each case.

While the event hold is on, Amazon S3 computes the retain-until-date as the current time plus the duration. The date moves forward as time passes. For example, suppose that the event hold duration is 1 year. If you retrieve the retain-until-date when the current time is `2027-01-01`, the date is `2028-01-01`. If you retrieve it again when the current time is `2027-02-01`, the date is `2028-02-01`.

![A timeline for a 1-year duration. When the current time is 2027-01-01, the retain-until-date is 2028-01-01. When the current time is 2027-02-01, it is 2028-02-01. The date moves forward as the current time advances.](https://docs.aws.amazon.com/AmazonS3/latest/userguide/images/object-lock-variable-retention-active_s3_timeline.png)


You can change the event hold duration. If you decrease it, Amazon S3 does not set the retain-until-date to an earlier date than its current value. For example, suppose that a 2-year duration set the retain-until-date to `2029-01-01`. If you decrease the duration to 1 year, the computed date is `2028-01-01`, but Amazon S3 doesn't apply it. The retain-until-date remains `2029-01-01`.

![A timeline showing that a 2-year duration sets the retain-until-date to 2029-01-01. Decreasing the duration to 1 year does not reduce it. A 1-year duration would compute 2028-01-01, but Amazon S3 does not apply that date. The retain-until-date remains 2029-01-01.](https://docs.aws.amazon.com/AmazonS3/latest/userguide/images/object-lock-variable-retention-decrease_s3_timeline.png)


When you release the event hold, Amazon S3 fixes the retain-until-date at the release time plus the duration, and the date stops changing. For example, with a 1-year duration, releasing the event hold on `2027-01-01` fixes the retain-until-date at `2028-01-01`. Until that date, the object version remains protected. After that date, you can delete it.

![A timeline showing two phases. While the hold is on, the retain-until-date is dynamic. When the hold is released, the date is fixed at the release time plus the duration. After that date, the object becomes deletable.](https://docs.aws.amazon.com/AmazonS3/latest/userguide/images/object-lock-variable-retention-release_s3_timeline.png)


If you specified a retain-until-date, the effective date is the later of that minimum date and the release time plus the duration. For example, suppose that the minimum retain-until-date is `2029-01-01` and the event hold duration is 1 year. If you release the event hold on `2028-06-01`, the computed date of `2029-06-01` is later than the minimum, so Amazon S3 applies it. If you release the event hold on `2027-01-01`, the computed date of `2028-01-01` is earlier than the minimum, so Amazon S3 applies the minimum date of `2029-01-01`.

![A comparison of two cases. When the release time plus the duration is later than the minimum retain-until-date, that later date applies. When it is earlier, the minimum retain-until-date applies.](https://docs.aws.amazon.com/AmazonS3/latest/userguide/images/object-lock-variable-retention-minimum_s3_timeline.png)


### Legal holds
<a name="object-lock-legal-holds"></a>

With Object Lock, you can also place a *legal hold* on an object version. Like a retention period, a legal hold prevents an object version from being overwritten or deleted. However, a legal hold doesn't have an associated fixed amount of time and remains in effect until removed. Legal holds can be freely placed and removed by any user who has the `s3:PutObjectLegalHold` permission. 

Legal holds are independent from retention periods. Placing a legal hold on an object version doesn't affect the retention mode or retention period for that object version. 

For example, suppose that you place a legal hold on an object version and that object version is also protected by a retention period. If the retention period expires, the object doesn't lose its WORM protection. Rather, the legal hold continues to protect the object until an authorized user explicitly removes the legal hold. Similarly, if you remove a legal hold while an object version has a retention period in effect, the object version remains protected until the retention period expires.

### How deletes work with S3 Object Lock
<a name="object-lock-how-deletes-work"></a>

If your bucket has S3 Object Lock enabled and the object is protected by a retention period or legal hold and you try to delete an object, Amazon S3 returns one of the following responses, depending on how you tried to delete the object:
+ **Permanent `DELETE` request** – If you issued a permanent `DELETE` request (a request that specifies a version ID), Amazon S3 returns an Access Denied (`403 Forbidden`) error when you try to delete the object. For more information about troubleshooting Access Denied errors with Object Lock, see [S3 Object Lock settings](troubleshoot-403-errors.md#troubleshoot-403-object-lock).
+ **Simple `DELETE` request** – If you issued a simple `DELETE` request (a request that doesn't specify a version ID), Amazon S3 returns a `200 OK` response and inserts a [delete marker](DeleteMarker.md) in the bucket, and that marker becomes the current version of the object with a new ID. For more information about managing delete markers with Object Lock, see [Managing delete markers with Object Lock](object-lock-managing.md#object-lock-managing-delete-markers).

### Best practices for using S3 Object Lock
<a name="object-lock-best-practices"></a>

Consider using *Governance mode* if you want to protect objects from being deleted by most users during a pre-defined retention period, but at the same time want some users with special permissions to have the flexibility to alter the retention settings or delete the objects. 

Consider using *Compliance mode* if you never want any user, including the root user in your AWS account, to be able to delete the objects during a pre-defined retention period. You can use this mode in case you have a requirement to store compliant data. 

You can use *Legal Hold* when you are not sure for how long you want your objects to stay immutable. This could be because you have an upcoming external audit of your data and want to keep objects immutable till the audit is complete. Alternately, you may have an ongoing project utilizing a dataset that you want to keep immutable until the project is complete. 

Consider using *variable retention* when you need to protect objects but don't know the retain-until-date at write time. For example, use it when you want a configurable recovery window against ransomware or accidental deletion. You can also use it when your retention period must begin on a future business event, such as contract completion, account closure, or claim resolution.

### Required permissions
<a name="object-lock-permissions"></a>

Object Lock operations require specific permissions. Depending on the exact operation that you're attempting, you might need any of the following permissions:
+ `s3:BypassGovernanceRetention`
+ `s3:GetBucketObjectLockConfiguration`
+ `s3:GetObjectLegalHold`
+ `s3:GetObjectRetention`
+ `s3:PutBucketObjectLockConfiguration`
+ `s3:PutObjectLegalHold`
+ `s3:PutObjectRetention`

For a complete list of Amazon S3 permissions with descriptions, see [ Actions, resources, and condition keys for Amazon S3](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazons3.html) in the *Service Authorization Reference*.

For more information about the permissions to S3 API operations by S3 resource types, see [Required permissions for Amazon S3 API operations](using-with-s3-policy-actions.md).

For information about using conditions with permissions, see [Bucket policy examples using condition keys](amazon-s3-policy-keys.md).