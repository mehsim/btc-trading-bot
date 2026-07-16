# AWS Running Costs Log
*Last Updated: 2026-07-16 13:28:35 UTC (Local: 18:28:35)*

---

## 1. Rate Schedule (AWS Tokyo - `ap-northeast-1`)

| Component | AWS Resource | Unit Cost | Daily Cost (Est.) |
| :--- | :--- | :--- | :--- |
| **Compute** | EC2 `t3.micro` (Linux) | `$0.0136 / hour` | `$0.3264` |
| **Storage** | 8 GB EBS SSD (`gp3` root) | `$0.0800 / GB-month` | `$0.0213` |
| **Network** | Elastic IP & Data Transfer | Free Tier / Negligible | `$0.0000` |

* **Total Daily Idle Cost**: **`$0.3477`**
* **Total Monthly Idle Cost**: **`$10.4300`**

---

## 2. Activity & Deployments Log

| Date (UTC) | Commit Hash / Action | Duration | Compute Cost | Storage Cost | Total Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-07-16 11:27:33** | Initial Mainnet Live Launch (`Jul 16, 11:27:33 UTC`) | Baseline Start | - | - | - |
| **2026-07-16 12:14:16** | Upgrade: WebSocket Orders & Redis Caching | 46 mins | `$0.0104` | `$0.0007` | `$0.0111` |
| **2026-07-16 12:30:35** | Upgrade: VTS Indicators & Adversarial Validation | 16 mins | `$0.0036` | `$0.0002` | `$0.0038` |
| **2026-07-16 13:03:30** | Recovery: Reclaimed Swap, Enabled 1GB Swap Memory | 33 mins | `$0.0075` | `$0.0005` | `$0.0080` |
| **2026-07-16 13:28:35** | Added `aws_cost.md` Tracker | 25 mins | `$0.0057` | `$0.0004` | `$0.0061` |

---

## 3. Cumulative Running Totals

* **Total Active Duration**: **1 hour, 56 minutes** (`1.93 hours`)
* **Cumulative Compute Cost**: **`$0.0263`**
* **Cumulative Storage Cost**: **`$0.0017`**
* **Total AWS Expenses to Date**: **`$0.0280`** (less than **3 cents**)
