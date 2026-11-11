

# IAM roles
<a name="id_roles"></a>

An IAM *role* is an IAM identity that you can create in your account that has specific permissions. An IAM role is similar to an IAM user, in that it is an AWS identity with permission policies that determine what the identity can and cannot do in AWS. However, instead of being uniquely associated with one person, a role is intended to be assumable by anyone who needs it. Also, a role does not have standard long-term credentials such as a password or access keys associated with it. Instead, when you assume a role, it provides you with temporary security credentials for your role session.

You can use roles to delegate access to users, applications, or services that don't normally have access to your AWS resources. For example, you might want to grant users in your AWS account access to resources they don't usually have, or grant users in one AWS account access to resources in another account. Or you might want to allow a mobile app to use AWS resources, but not want to embed AWS keys within the app (where they can be difficult to update and where users can potentially extract them). Sometimes you want to give AWS access to users who already have identities defined outside of AWS, such as in your corporate directory. Or, you might want to grant access to your account to third parties so that they can perform an audit on your resources.

For these scenarios, you can delegate access to AWS resources using an *IAM role*. This section introduces roles and the different ways you can use them, when and how to choose among approaches, and how to create, manage, switch to (or assume), and delete roles.

[Account access manager](account-access-manager.md) is an IAM feature that lets you centrally assign IAM roles across your organization's accounts to [IAM Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html) users and groups. Account access manager gives your workforce access through IAM roles with their full feature set. You can use it alongside IAM Identity Center permission sets or on its own.

**Note**  
When you first create your AWS account, no roles are created by default. As you add services to your account, they may add service-linked roles to support their use cases.  
 A service-linked role is a type of service role that is linked to an AWS service. The service can assume the role to perform an action on your behalf. Service-linked roles appear in your AWS account and are owned by the service. An IAM administrator can view, but not edit the permissions for service-linked roles.   
Before you can delete service-linked roles you must first delete their related resources. This protects your resources because you can't inadvertently remove permission to access the resources.  
For information about which services support using service-linked roles, see [AWS services that work with IAM](reference_aws-services-that-work-with-iam.md) and look for the services that have **Yes **in the **Service-Linked Role** column. Choose a **Yes** with a link to view the service-linked role documentation for that service.

**Topics**
+ [When to create an IAM user (instead of a role)](#id_which-to-choose)
+ [Roles terms and concepts](#id_roles_terms-and-concepts)
+ [Additional resources](#id_roles_additional-resources)
+ [The confused deputy problem](confused-deputy.md)
+ [Common scenarios for IAM roles](id_roles_common-scenarios.md)
+ [IAM role creation](id_roles_create.md)
+ [IAM role management](id_roles_manage.md)
+ [Methods to assume a role](id_roles_manage-assume.md)

## When to create an IAM user (instead of a role)
<a name="id_which-to-choose"></a>

We recommend you only use IAM users for use cases not supported by identity federation. Some of the use cases include the following:
+ **Workloads that cannot use IAM roles** – You might run a workload from a location that needs to access AWS. In some situations, you can't use IAM roles to provide temporary credentials, such as for WordPress plugins. In these situations, use IAM user long-term access keys for that workload to authenticate to AWS.
+ **Third-party AWS clients** – If you are using tools that don’t support access with IAM Identity Center, such as third-party AWS clients or vendors that aren't hosted on AWS, use IAM user long-term access keys.
+ **AWS CodeCommit access** – If you are using CodeCommit to store your code, you can use an IAM user with either SSH keys or service-specific credentials for CodeCommit to authenticate to your repositories. We recommend that you do this in addition to using a user in IAM Identity Center for normal authentication. Users in IAM Identity Center are the people in your workforce who need access to your AWS accounts or to your cloud applications. To give users access to your CodeCommit repositories without configuring IAM users, you can configure the **git-remote-codecommit** utility. For more information about IAM and CodeCommit, see [IAM credentials for CodeCommit: Git credentials, SSH keys, and AWS access keys](id_credentials_ssh-keys.md). For more information about configuring the **git-remote-codecommit** utility, see [Connecting to AWS CodeCommit repositories with rotating credentials](https://docs.aws.amazon.com/codecommit/latest/userguide/temporary-access.html#temporary-access-configure-credentials) in the *AWS CodeCommit User Guide*.
+ **Amazon Keyspaces (for Apache Cassandra) access** – In a situation where you are unable to use users in IAM Identity Center, such as for testing purposes for Cassandra compatibility, you can use an IAM user with service-specific credentials to authenticate with Amazon Keyspaces. Users in IAM Identity Center are the people in your workforce who need access to your AWS accounts or to your cloud applications. You can also connect to Amazon Keyspaces using temporary credentials. For more information, see [Using temporary credentials to connect to Amazon Keyspaces using an IAM role and the SigV4 plugin](https://docs.aws.amazon.com/keyspaces/latest/devguide/access.credentials.html#temporary.credentials.IAM) in the *Amazon Keyspaces (for Apache Cassandra) Developer Guide*.
+ **Emergency access** – In a situation where you can't access your identity provider and you must take action in your AWS account. Establishing emergency access IAM users can be part of your resiliency plan. We recommend that the emergency user credentials be tightly controlled and secured using multi-factor authentication (MFA).

## Roles terms and concepts
<a name="id_roles_terms-and-concepts"></a>

Here are some basic terms to help you get started with roles.

****Role****  
An IAM identity that you can create in your account that has specific permissions. An IAM role has some similarities to an IAM user. Roles and users are both AWS identities with permissions policies that determine what the identity can and cannot do in AWS. However, instead of being uniquely associated with one person, a role is intended to be assumable by anyone who needs it. Also, a role does not have standard long-term credentials such as a password or access keys associated with it. Instead, when you assume a role, it provides you with temporary security credentials for your role session.  
Roles can be assumed by the following:  
+ An IAM user in the same AWS account or another AWS account
+ IAM roles in the same account
+ Service principals, for use with AWS services and features like:
  + Services that allow you to run code on compute services, like Amazon EC2 or AWS Lambda
  + Features that perform actions to your resources on your behalf, like Amazon S3 object replication
  + Services that deliver temporary security credentials to your applications that run outside of AWS, such as IAM Roles Anywhere or Amazon ECS Anywhere
+ An external user authenticated by an external identity provider (IdP) service that is compatible with SAML 2.0 or OpenID Connect

****AWS service role****  
 A service role is an [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) that a service assumes to perform actions on your behalf. An IAM administrator can create, modify, and delete a service role from within IAM. For more information, see [Create a role to delegate permissions to an AWS service](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-service.html) in the *IAM User Guide*. 

****AWS service-linked role****  
 A service-linked role is a type of service role that is linked to an AWS service. The service can assume the role to perform an action on your behalf. Service-linked roles appear in your AWS account and are owned by the service. An IAM administrator can view, but not edit the permissions for service-linked roles.   
If you are already using a service when it begins supporting service-linked roles, you might receive an email announcing a new role in your account. In this case, the service automatically created the service-linked role in your account. You don't need to take any action to support this role, and you should not manually delete it. For more information, see [A new role appeared in my AWS account](troubleshoot_roles.md#troubleshoot_roles_new-role-appeared).
For information about which services support using service-linked roles, see [AWS services that work with IAM](reference_aws-services-that-work-with-iam.md) and look for the services that have **Yes **in the **Service-Linked Role** column. Choose a **Yes** with a link to view the service-linked role documentation for that service. For more information, see [Create a service-linked role](id_roles_create-service-linked-role.md).

****Role chaining****  
Role chaining is when you use a role to assume a second role. You can perform role chaining through the AWS Management Console by switching roles, the AWS CLI, or API. For example, `RoleA` has permission to assume `RoleB`. You can enable User1 to assume `RoleA` by using their long-term user credentials in the AssumeRole API operation. This returns `RoleA` short-term credentials. With role chaining, you can use `RoleA`'s short-term credentials to enable User1 to assume `RoleB`.  
When you assume a role, you can pass a session tag and set the tag as transitive. Transitive session tags are passed to all subsequent sessions in a role chain. To learn more about session tags, see [Pass session tags in AWS STS](id_session-tags.md).  
Role chaining limits your AWS Management Console, AWS CLI or AWS API role session to a maximum of one hour. It applies regardless of the maximum session duration configured for individual roles. When you use the [AssumeRole](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) API operation to assume a role, you can specify the duration of your role session with the `DurationSeconds` parameter. You can specify a parameter value of up to 43200 seconds (12 hours), depending on the [maximum session duration setting](id_roles_update-role-settings.md#id_roles_update-session-duration) for your role. However, if you assume a role using role chaining and provide a `DurationSeconds` parameter value greater than one hour, the operation fails.  
For information about switching to a role in the AWS Management Console, see [Switch from a user to an IAM role (console)](id_roles_use_switch-role-console.md).

****Delegation****  
The granting of permissions to someone to allow access to resources that you control. Delegation involves setting up a trust between two accounts. The first is the account that owns the resource (the trusting account). The second is the account that contains the users that need to access the resource (the trusted account). The trusted and trusting accounts can be any of the following:  
+ The same account.
+ Separate accounts that are both under your organization's control.
+ Two accounts owned by different organizations.
To delegate permission to access a resource, you [create an IAM role](id_roles_create_for-user.md) in the trusting account that has two policies attached. The *permissions policy* grants the user of the role the needed permissions to carry out the intended tasks on the resource. The *trust policy* specifies which trusted account members are allowed to assume the role.  
When you create a trust policy, you cannot specify a wildcard (\*) as part of an ARN in the principal element. The trust policy is attached to the role in the trusting account, and is one-half of the permissions. The other half is a permissions policy attached to the user in the trusted account that [allows that user to switch to, or assume the role](id_roles_use_permissions-to-switch.md). A user who assumes a role temporarily gives up his or her own permissions and instead takes on the permissions of the role. When the user exits, or stops using the role, the original user permissions are restored. An additional parameter called [external ID](id_roles_common-scenarios_third-party.md#id_roles_third-party_external-id) helps ensure secure use of roles between accounts that are not controlled by the same organization.

****Trust policy****  
A [JSON policy document](reference_policies_grammar.md) in which you define the principals that you *trust* to assume the role. A role trust policy is a required [resource-based policy](access_policies.md#policies_resource-based) that is attached to a role in IAM. The [principals](reference_policies_elements_principal.md) that you can specify in the trust policy include users, roles, accounts, and services. For more information, see [How to use trust policies in IAM roles](https://aws.amazon.com/blogs/security/how-to-use-trust-policies-with-iam-roles/) in *AWS Security Blog*.

****Role for cross-account access****  
A role that grants access to resources in one account to a trusted principal in a different account. Roles are the primary way to grant cross-account access. However, some AWS services allow you to attach a policy directly to a resource (instead of using a role as a proxy). These are called resource-based policies, and you can use them to grant principals in another AWS account access to the resource. Some of these resources include Amazon Simple Storage Service (S3) buckets, Amazon Glacier vaults, Amazon Simple Notification Service (SNS) topics, and Amazon Simple Queue Service (SQS) queues. To learn which services support resource-based policies, see [AWS services that work with IAM](reference_aws-services-that-work-with-iam.md). For more information about resource-based policies, see [Cross account resource access in IAM](access_policies-cross-account-resource-access.md).

## Additional resources
<a name="id_roles_additional-resources"></a>

The following resources can help you learn more about IAM terminology related to IAM roles.
+ **Principals** are entities in AWS that can perform actions and access resources. A principal can be an AWS account root user, an IAM user, or a role. A principal that represents the identity of an AWS service is a [service principal](reference_policies_elements_principal.md#principal-services). Use the Principal element in role trust policies to define the principals that you trust to assume the role.

   For more information and examples of principals you can allow to assume a role, see [AWS JSON policy elements: Principal](reference_policies_elements_principal.md). 
+ **Identity federation** creates a trust relationship between an external identity provider and AWS. You can use your existing OpenID Connect (OIDC) or Security Assertion Markup Language (SAML) 2.0 provider to manage who can access AWS resources. When you use OIDC and SAML 2.0 to configure a trust relationship between these external identity providers and AWS , the user is assigned to an IAM role. The user also receives temporary credentials that allow the user to access your AWS resources.

  For more information about federated principals, see [Identity providers and federation into AWS](id_roles_providers.md).
+ **Federated principals** are existing identities from Directory Service, your enterprise user directory, or an OIDC provider. AWS assigns a role to a federated principal when access is requested through an [identity provider](id_roles_providers.md).

  For more information about SAML and OIDC federated principals, see [Federated user sessions and roles](introduction_access-management.md#intro-access-roles).
+ **Permissions policies** are identity-based policies that define what actions and resources the role can use. The document is written according to the rules of the IAM policy language. 

  For more information, see [IAM JSON policy reference](reference_policies.md).
+ **Permissions boundaries** are an advanced feature in which you use policies to limit the maximum permissions that an identity-based policy can grant to a role. You cannot apply a permissions boundary to a service-linked role.

  For more information, see [Permissions boundaries for IAM entities](access_policies_boundaries.md).