

# Amazon EC2 security groups for your EC2 instances
<a name="ec2-security-groups"></a>

A *security group* acts as a virtual firewall for your EC2 instances to control incoming and outgoing traffic. Inbound rules control the incoming traffic to your instance, and outbound rules control the outgoing traffic from your instance. When you launch an instance, you can specify one or more security groups. If you don't specify a security group, Amazon EC2 uses the default security group for the VPC. After you launch an instance, you can change its security groups.

Security is a shared responsibility between AWS and you. For more information, see [Security in Amazon EC2](ec2-security.md). AWS provides security groups as one of the tools for securing your instances, and you need to configure them to meet your security needs. If you have requirements that aren't fully met by security groups, you can maintain your own firewall on any of your instances in addition to using security groups.

**Pricing**  
There is no additional charge for using security groups.

**Topics**
+ [Overview](#security-group-basics)
+ [Create a security group for your Amazon EC2 instance](creating-security-group.md)
+ [Change the security groups for your Amazon EC2 instance](changing-security-group.md)
+ [Delete an Amazon EC2 security group](deleting-security-group.md)
+ [Amazon EC2 security group connection tracking](security-group-connection-tracking.md)
+ [Security group rules for different use cases](security-group-rules-reference.md)

## Overview
<a name="security-group-basics"></a>

You can associate each instance with multiple security groups, and you can associate each security group with multiple instances. You add rules to each security group that allow traffic to or from its associated instances. You can modify the rules for a security group at any time. New and modified rules are automatically applied to all instances that are associated with the security group. When Amazon EC2 decides whether to allow traffic to reach an instance, it evaluates all rules from all security groups that are associated with the instance. For more information, see [Security group rules](https://docs.aws.amazon.com/vpc/latest/userguide/security-group-rules.html) in the *Amazon VPC User Guide*.

The following diagram shows a VPC with a subnet, an internet gateway, and a security group. The subnet contains EC2 instances. The security group is associated with the instances. The only traffic that reaches the instance is the traffic allowed by the security group rules. For example, if the security group contains a rule that allows SSH traffic from your network, then you can connect to your instance from your computer using SSH. If the security group contains a rule that allows all traffic from the resources associated with it, then each instance can receive any traffic sent from the other instances.

![VPC with a security group. The EC2 instances in the subnet are associated with the security group.](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/images/ec2-security-groups.png)


Security groups are stateful—if you send a request from your instance, the response traffic for that request is allowed to flow in regardless of inbound security group rules. Also, responses to allowed inbound traffic are allowed to flow out, regardless of outbound rules. For more information, see [Connection tracking](security-group-connection-tracking.md).