# NegotiateEnv - 3 Minute Demo Script
## OpenEnv Hackathon Submission

**Team**: Madhavi Gulavani, Mayuka Reddy, Kushal Adhyaru

---

## [0:00-0:30] Introduction & Problem Statement

"Hi! We built **NegotiateEnv** - an OpenEnv-compatible RL environment where AI agents learn to negotiate B2B SaaS contracts.

**Why negotiation?** Every current OpenEnv environment - Wordle, Sudoku, Blackjack - has fully visible state and frequent rewards. Negotiation breaks both:
- The vendor's floor price is **always hidden**
- Rewards only come at deal close or walkaway
- A rule-based opponent with 4 distinct strategies
- Mid-episode constraint drift that forces strategy adaptation

This tests **hidden-information multi-turn reasoning** - a capability no current OpenEnv environment covers."

---

## [0:30-1:15] Live Demo - Before Training

**[SCREEN: Terminal with demo.py]**

"Let me show you what happens before training. This is our rule-based baseline agent."

```bash
python demo.py --env-url https://kushaladhyaru-negotiate-env.hf.space
```

**[Point to output]**

"The agent sees:
- Vendor's opening: $175/seat, 3-year contract
- Budget constraint: $148/seat max
- But the vendor's floor price? **Hidden**

Watch what happens..."

**[Show transcript]**
- Turn 1: Agent probes for information
- Turn 2: Agent counters at budget price
- Turn 3: Vendor responds with concession
- Turn 4: Deal closes

"Final reward: **0.48** - decent, but the agent is leaving money on the table."

---

## [1:15-2:00] Training Architecture & Results

**[SCREEN: Switch to colab_training_full.ipynb or show reward curve]**

"We trained using:
- **Dataset**: 200 synthetic B2B scenarios on HuggingFace
- **Model**: Qwen2.5-1.5B-Instruct with 4-bit LoRA
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Training**: 500 episodes on Colab H100

Here's the reward curve..."

**[Show reward_curve.png if available]**

"Starting reward: **0.15** - basically random
Final reward: **0.62** - that's a **4x improvement**

More importantly, look at the strategy evolution:
- Before: 52% early accepts, 12% counteroffers
- After: 48% counteroffers, 22% probing, 14% accepts

The agent **learned real negotiation tactics**."

---

## [2:00-2:30] Key Innovation - Multi-Agent Interactions

**[SCREEN: Show code or architecture diagram]**

"What makes this unique for the hackathon?

**1. Partial Observability**: The vendor's floor price is never exposed - the agent must infer it through negotiation.

**2. Adversarial Opponent**: Four distinct AE strategies:
   - Hardball: Minimal concessions, false 'final offers'
   - Concession trader: Only moves if you extend contract
   - Urgency: Time pressure tactics
   - Cooperative: Genuinely seeks middle ground

**3. Constraint Drift**: Mid-negotiation, constraints change:
   - 'Budget cut 10% - CFO reduced spend'
   - 'Acquisition added 20 designers'
   
This forces real-time strategy adaptation."

---

## [2:30-3:00] Deployment & Submission

**[SCREEN: Show links]**

"Everything is live and ready:

✅ **Environment**: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env
✅ **Dataset**: https://huggingface.co/datasets/mayukareddy/SyntheticSaasDataset  
✅ **Trained Model**: https://huggingface.co/KushalAdhyaru/negotiate-env-qwen-unsloth-500ep
✅ **Code**: https://github.com/MadhaviSG/openEnv-negotiateEnv

You can test it right now:"

```bash
curl https://kushaladhyaru-negotiate-env.hf.space/health
```

"**NegotiateEnv** - teaching AI agents to negotiate like humans, one deal at a time.

Thank you!"

---

## Quick Test Commands (Backup)

If you need to show live API calls:

```bash
# Health check
curl https://kushaladhyaru-negotiate-env.hf.space/health

# Start negotiation
curl -X POST https://kushaladhyaru-negotiate-env.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'

# Make counteroffer
curl -X POST https://kushaladhyaru-negotiate-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID_FROM_RESET",
    "action": {
      "action_type": "counter",
      "price_per_seat": 148,
      "contract_length": 2,
      "annual_increase_cap": 3,
      "message": "Our budget is $148/seat max"
    }
  }'
```

---

## Timing Breakdown

- **0:00-0:30**: Problem statement (30s)
- **0:30-1:15**: Live demo before training (45s)
- **1:15-2:00**: Training results & strategy evolution (45s)
- **2:00-2:30**: Key innovations (30s)
- **2:30-3:00**: Deployment & wrap-up (30s)

**Total: 3:00**

---

## Visual Aids to Prepare

1. Terminal with `demo.py` ready to run
2. Reward curve image (`reward_curve.png`)
3. Browser tabs open to:
   - HuggingFace Space
   - Dataset page
   - Model page
   - GitHub repo
4. Comparison table (before/after training)

---

## Key Talking Points

✅ **Only OpenEnv environment with hidden information**  
✅ **Multi-agent with adversarial opponent**  
✅ **Real-world professional task (B2B procurement)**  
✅ **4x reward improvement through training**  
✅ **Strategy emergence: learned to probe, counter, leverage**  
✅ **Fully deployed and accessible**

---

## Backup Slides (If Demo Fails)

Have screenshots ready of:
- Successful negotiation transcript
- Reward curve showing 0.15 → 0.62
- Strategy distribution table
- Live Space running

---

## Questions You Might Get

**Q: Why negotiation vs other environments?**  
A: It's the only OpenEnv task with hidden information and adversarial opponents - tests theory-of-mind reasoning.

**Q: How do you prevent the agent from just accepting everything?**  
A: Reward function penalizes bad deals. Walking away from overpriced contracts gives +0.1, accepting them gives -0.2.

**Q: Can it handle real negotiations?**  
A: The scenarios are synthetic but based on real B2B SaaS pricing. The constraint drift (budget cuts, scope changes) mirrors real procurement workflows.

**Q: What's the training cost?**  
A: ~40 minutes on Colab H100 (free tier). We also provide a T4-compatible version with Unsloth.

---

Good luck with your demo! 🚀
