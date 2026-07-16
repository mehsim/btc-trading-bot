# AWS Running Costs Log
*Last Updated: 2026-07-16 16:47:00 UTC (Local: 21:47:00)*

---

## 1. Rate Schedule (AWS Tokyo - `ap-northeast-1`)

| Component   | AWS Resource              | Unit Cost             | Daily Cost (Est.) |
| :---        | :---                      | :---                  | :---              |
| **Compute** | EC2 `t3.micro` (Linux)    | `$0.0136 / hour`      | `$0.3264`         |
| **Storage** | 8 GB EBS SSD (`gp3` root) | `$0.0800 / GB-month`   | `$0.0213`         |
| **Network** | Elastic IP & Data Transfer| Free Tier / Negligible| `$0.0000`         |

* **Total Daily Idle Cost**: **`$0.3477`**
* **Total Monthly Idle Cost**: **`$10.4300`**

---

## 2. Activity & Deployments Log

| Date (UTC)             | Commit Hash / Action                                    | Duration  | Compute Cost | Storage Cost | Total Cost |
| :---                   | :---                                                    | :---      | :---         | :---         | :---       |
| **2026-07-16 11:27:33** | Initial Mainnet Live Launch (`Jul 16, 11:27:33 UTC`)    | Baseline  | -            | -            | -          |
| **2026-07-16 12:14:16** | Upgrade: WebSocket Orders & Redis Caching                | 46 mins   | `$0.0104`    | `$0.0007`    | `$0.0111`  |
| **2026-07-16 12:30:35** | Upgrade: VTS Indicators & Adversarial Validation        | 16 mins   | `$0.0036`    | `$0.0002`    | `$0.0038`  |
| **2026-07-16 13:03:30** | Recovery: Reclaimed Swap, Enabled 1GB Swap Memory       | 33 mins   | `$0.0075`    | `$0.0005`    | `$0.0080`  |
| **2026-07-16 13:28:35** | Added `aws_cost.md` Tracker                              | 25 mins   | `$0.0057`    | `$0.0004`    | `$0.0061`  |
| **2026-07-16 16:45:00** | Recovery: Expanded Swap to 2GB, trained 60m/120m models | 196 mins  | `$0.0445`    | `$0.0029`    | `$0.0474`  |
| **2026-07-16 16:55:00**| Bugfix: Fixed ensemble multi_class error for stacking   | 10 mins   | `$0.0023`    | `$0.0001`    | `$0.0024`  |
| **2026-07-16 17:00:00**| Deployment: Enabled 24-hour trading session              | 5 mins    | `$0.0011`    | `$0.0001`    | `$0.0012`  |
| **2026-07-16 17:38:00**| Recovery: Trained 240m strategy models and restarted bot | 39 mins   | `$0.0088`    | `$0.0006`    | `$0.0094`  |

---

## 3. Cumulative Running Totals

* **Total Active Duration**: **6 hours, 6 minutes** (`6.10 hours`)
* **Cumulative Compute Cost**: **`$0.0830`**
* **Cumulative Storage Cost**: **`$0.0054`**
* **Total AWS Expenses to Date**: **`$0.0884`** (approx. **8.8 cents**)
