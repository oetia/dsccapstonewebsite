# Multi-Modal LLM Reasoning and Agent Modeling

Trevan Nguyen, Nathaniel Del Rosario, Aaryan Agrawal, Zihan Liu, Samuel Zhang, Zhiting Hu

## Introduction 

Web-based agents using LLMs show promise in automating browser tasks,
but scaling inference efficiently remains a challenge. This work explores the
question of how best to structure search: implicit (greedy, depth-limited) or
explicit (structured exploration like MCTS). Implicit search is potentially com-
putationally cheaper but struggles with backtracking, while explicit search
enables efficient exploration but relies on resettable states, which may be im-
practical in real-world web environments. Experiments on 106 WebArena
tasks show explicit search achieves higher task completion rates and better
environment interaction efficiency. While explicit search excels in controlled
settings, implicit search remains more applicable to real-world tasks. An-
other aspect to consider is conducting an explicit search on an LLM world
model, where the search occurs over predicted next states as opposed to the
environment itself, which can potentially gain the benefits of both implicit
and explicit search. These techniques extend beyond web environments, and
should theoretically be applicable to OS automation (OsWorld) and dynamic
game environments (MineDojo).

## Methods

### Browsergym

Browsergym \cite{chezelles2024browsergym} is the primary environment that we are focusing on. Browsergym essentially provides an OpenAI gym-like environment \cite{brockman2016openai} for the web browser. The env object takes in an action represented as code and provides an observation at each step. By utilizing the browsergym library, we can test the performance of our web agent on two key browser task benchmarks: \textbf{WebArena} \cite{zhou2023webarena} and \textbf{Assistantbench} \cite{yoran2024assistantbench}.

### Actions

How the action is represented is something that can be slightly finicky. Since browsergym is built off of playwright, the action is going to eventually be JavaScript code that is executed to interact with the DOM. The space of all possible JavaScript code is a massive action space, and allowing an agent to directly interact with this space creates an correspondingly massive search space. Setting aside issues with search complexity and code correctness, fundamentally all of the tasks that such an agent would be expected to solve, would be doable with the action space available to the average human end user, i.e. just a keyboard and mouse. 

For this reason, functions have been predefined for actions such as `click`, `fill`, `go\_back`, `go\_forward`, etc. While the environment can accept arbitrary code, the agent has been instructed to only provide a specific set of function calls constrained to a "human" action space. 

### Observations

After an action is used to step the environment, an observation is returned. This observation is also provided upon the environment instantiation. By default it contains the page HTML, AXTree, and a screenshot of the current state. Directly passing in all of this information into the context of an LLM, especially the HTML, seems to lead to a significant amount of noise, and degrades performance. For a webpage such as Reddit, the HTML you'd get from the homepage can easily be hundreds or even thousands of lines long. If the first step of your task is just to use the search bar to look for a specific subreddit, 99\%+ of the elements will be irrelevant. The same to some extent also applies to the accessibility tree, however, the AXTree being a significantly more compact representation, takes up significantly less context. Only the AXTree ends up being passed into LLM context. 

For screenshots, since current LLMs tend to struggle with grounding, the screenshots are further augmented with a set of marks (SoM) \cite{yang2023set}. 

### Search

At every step, a single action can be taken to expand into a new state. For a given task, there can often be over a dozen steps needed to reach completion. If there's a situation where the first ten steps are correct, but a minor mistake is made on the eleventh step, the task would become in-completable without backtracking. 

### Implicit Search on the Environment

You could backtrack by relying on the LLM agent to undo it's action, i.e. if a subscribe button is clicked, it would then click the unsubscribe button to undo it. However, if the LLM clicked a button which brings up a modal form, where it's still on the same page, then to close the modal, the LLM sends the go\_back action, which navigates to the previous page, then while it has closed the modal, it has gone back too far and failed it's backtrack. While such scenarios should be recoverable, empirically speaking, the LLM struggles to do so, and task execution becomes messy.


### Explicit Search on the Environment

An alternative would be relying on a search algorithm, such as MCTS, to do backtracking for you. In an arbitrary environment, this could involve resetting the environment, then replaying all actions, but in the web case, you can do something more sophisticated with caching and reloading web pages. Having an explicit search algorithm like MCTS also provides other benefits in that an LLM doesn't need to identify the correct backtrack and subsequent next node to end up finding the correct trajectory. 

Under an explicit search algorithm like MCTS, at every step when expanding a node, the LLM is re-prompted to generate hundreds of possible next actions, i.e. new nodes. Then another LLM can generate an evaluation of each action, so that the rollout isn't random. These evaluations can then be used to influence the subsequent Q-values and guide exploration. 

It should be noted that a strong assumption is being made here that backtracking would be possible. In any situation where you are writing information to some external server that you don't control, you run the risk of a backtrack failing. Simply reloading a cached client state will not reset the server state. If a bank transfer is made on something like Zelle, and then you want to backtrack from that state, you cannot do so. 

This does make implicitly searching on the environment preferable to explicitly doing so in many cases, as there is no dependence on such an assisted backtrack. Comparing the performance of these two would be interesting. 


### Comparing Explicit and Implicit Environment Search


Implicit search is essentially greedy search, and greedy search can be represented through MCTS with only a single iteration. Visitation statistics do not accumulate, so the only information being used to decide which node to expand next is the LLM evaluation of an action (fast reward). Implicit search will be implemented as MCTS(depth=..., iterations=1).

When it comes to comparing explicit search and implicit search and how they scale, implicit search can only be scaled by increasing the depth. A nice property is that when running to evaluate performance on a depth 100 implicit search, you can just ignore the later portions of trajectories to get results for all depths before 100. For gathering data on how implicit search scales, a single run of "MCTS" at MCTS(depth=100, iterations=1) will be sufficient.

For explicit search, there does not exist this same property for depth. If you have a 10 iteration 20 depth tree result for a task, if for the first iteration it reaches task completion on depth 15, then the search would end leading to no second iteration. If you try to extrapolate a 10 iteration 10 depth run from this data, you would not be able to do so. However, that does not mean that a 10 iteration 10 depth tree is incapable of also finding a viable solution. You have to do a separate run to find out. 

However, for explicit search and iterations, such a convenient property does exist. Through ignoring the later iterations a 10 iteration 20 depth experiment would inherently provide a run for a 9 iteration 20 depth experiment and so on so forth. 

With these bits of information in mind, for explicit search, there will be separate runs which will all share the same number of iterations at 10, but have varying depths of 5, 10, and 20. 

In the context of browsergym, the benchmark used for these experimental runs will be the webarena benchmark. Of over 800 tasks provided, a subset of 106 tasks will be evaluated on. The main reason for a subset fundamentally comes down to a matter of cost. When running a single example on gpt-4o with MCTS(depth=20, iterations=10), the cost can already easily exceed \$1 USD. Should you evaluate on the entire dataset with gpt-4o, a single run would likely take over \$1000. With 4 runs planned, some of them likely to be far more expensive than \$1 per task, the estimated cost of this entire experiment would likely be more than \$4000. 

With a provided key with \$200 of credit, not only is a subset needed, but also a cheaper model. These experiments will be conducted on said subset of 106 tasks, and also utilize gpt-4o-mini instead of gpt-4o, which should further reduce the costs by approximately 15 fold. The number of action proposals at each step will be kept at 10. While it could be set much higher, with a proposal temperature of 0.7, often over half of the proposals end up being duplicate actions. Scaling n proposals at each step is another axis and increasing it to a 100 or more would likely also benefit performance, but that can be explored later. For now, keeping n proposals fixed at 10 should provide enough variety in responses for benefits to be attainable from search. 

Despite gpt-4o-mini being a distill of gpt-4o, the scaling results may not necessarily generalize to gpt-4o, and should be taken with caution. 


### Search on an Internal World Model

Another alternative to addressing the backtracking issue in the explicit search on the environment case is to search and backtrack not on the actual environment, but instead on a simulated "LLM dream". On top of having a step function for the browsergym environment, you also have the LLM approximate the results on the step function. This addresses the issue of some actions being irreversible with the downside of becoming dependent on the LLM's ability to "dream" the browser environment accurately. As such, the LLM is considered an Internal (as opposed to the external, real browser environment) World Model (where our "world" is the browser environment).  

