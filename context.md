I want to create a new mobile app, in the 'food tracker' / 'macro tracker' category

my goal is to make it as convenient as possible to update the food diary

there's plenty of apps that let you take pictures and do AI detection, etc.

what I want to do is extend what apps already do today

- allow uploading _multiple_ pictures at once, to reduce the friction of potentially doing an upload once a day or sometimes even every couple days

- SAVE the pictures of the uploaded food against the entries

- if pictures are converted into ingredients lists and added that way - we should be able to modify the ingredients at a later time - not add it and its once and done (which is the case for popular apps like macrofactor)

- enable features like if we take pictures of food on a scale, it'll estimate the ingredients in a way that actually adds up to the shown weight

- have a branching agent - e.g. if we show a scale with weight, assume the weight both includes, and does not include, the container, etc.

please help research several thigns

- what features can help us differentiate

- what price range can be so competitve, assuming SEO, etc. is in place, how we can make it a winning choice for certain potential users

- in a lot of detail - how to implement the AI agent side of making food calorie estimation AS accurate as possible

- for now, let's try and use ADK for this

---

Comprehensive Strategic Framework for Multimodal Agentic Nutrition Tracking: Architectural Paradigms, Market Positioning, and Implementation of Frictionless Batch-Logging Systems

The landscape of digital dietary assessment in 2026 is undergoing a fundamental transition from manual record-keeping to autonomous, vision-led sensing. This shift is necessitated by the persistent failure of traditional tracking applications to solve the primary barrier to long-term adherence: logging friction. While incumbents like MyFitnessPal and Cronometer have dominated the market through database scale and clinical precision, a significant gap remains for a solution that prioritizes the temporal flexibility of the user. The emergence of Agentic AI—systems capable of perception, reasoning, and autonomous action—provides the necessary framework to move beyond simple image recognition toward complex, context-aware dietary monitoring.  
The Evolution of Dietary Monitoring and the Persistence of Logging Fatigue

Digital nutrition tracking has evolved through three distinct eras. The first era relied on self-reporting and manual entry, which, while foundational, introduced significant errors due to subject recall bias and the sheer burden of data entry. The second era introduced computer-assisted features, most notably barcode scanning and rudimentary image recognition, yet these remained "active" processes requiring real-time interaction at every meal. The current third era, Image-Assisted Dietary Assessment (IADA), leverages deep learning to identify foods and estimate portions, yet most commercial applications still follow a linear, transaction-based workflow: one meal, one photo, one entry.

The primary psychological obstacle to consistent tracking is not the lack of data but the "cost of interruption". Research into user sentiment indicates that tracking often feels like a full-time job, leading to "logging streaks" followed by complete abandonment once a user misses a single day. This "once and done" architecture, where entries are difficult to modify retrospectively and photos are treated as transient evidence rather than persistent data, fails to account for the reality of modern life. A system that allows for asynchronous batch uploading of multiple images addresses this by decoupling the act of eating from the act of logging, thereby reducing the cognitive load and making adherence sustainable over months rather than weeks.  
Table 1: Comparative Analysis of Logging Friction and Action Scores
Metric MyFitnessPal MacroFactor Cronometer Proposed "Batch" System
Actions for Search Log 15 steps 10 steps 12 steps < 8 steps
Actions for Barcode Scan 5 steps 2 steps 3 steps 2 steps
Actions for Photo Log 8 steps 4 steps 6 steps 3 steps (Average)
Batch Support No No Limited Yes (Up to 20 images)
Modification Style Destructive Semi-Persistent Static Fully Retrospective
Estimated Daily Time 7-10 minutes 3-5 minutes 5-8 minutes 1-2 minutes

Market Topology and Competitive Gap Analysis

To successfully differentiate in a saturated market, an application must solve the "unmet needs" of high-intent users who find existing tools either too simplistic or overly rigid. MacroFactor has established a leadership position in logging speed and adherence-neutral data processing, yet its AI photo recognition often lacks deep context, such as the ability to reconcile scale weight with visual volume. Conversely, Cronometer offers unparalleled micronutrient data but lacks the seamless multimodal experience that users in 2026 expect.  
Strategic Differentiation Factors

The proposed application identifies three critical pillars of differentiation:

    Temporal Flexibility (Batch Uploading): Existing apps typically force a real-time logging paradigm. By allowing the upload of multiple pictures from the camera roll simultaneously, the system accommodates the "catch-up" behavior typical of busy professionals.

    Persistent Evidence (Photo-Linked Entries): Unlike MacroFactor, where the link between the photo and the resulting ingredient list is often obscured after the initial log, the proposed system maintains the photo as a persistent artifact. This allows users to revisit an entry days later, see the original meal, and adjust the ingredients based on new information.

    Physical Calibration (Scale-Aware Branching): Current IADA systems struggle with volume estimation from 2D images. By integrating scale weight detection and a branching agent that tests hypotheses (e.g., container weight vs. net food weight), the system achieves a level of precision previously reserved for laboratory settings.

Table 2: Feature Maturity and User Segment Alignment
Feature Segment: Biohackers Segment: Busy Professionals Segment: Clinical/Medical Segment: Budget-Conscious
Batch Photo Upload Low Priority High Priority Moderate Priority Moderate Priority
Scale Weight AI High Priority Moderate Priority High Priority Low Priority
Ingredient Modifiability High Priority High Priority High Priority Moderate Priority
NUTTAB/USDA Depth High Priority Low Priority High Priority Low Priority
Multi-User/Family Plan Moderate Priority High Priority Moderate Priority High Priority

Monetization and Pricing Strategy for Competitive Disruption

In 2026, the Health and Fitness category remains one of the most lucrative app segments, with high-performing apps making over 400 times more revenue than the bottom quartile. However, this revenue is concentrated among apps that demonstrate immediate value and clear differentiation. For a new entrant, a "Competitive Disruption" pricing model is required to overcome the brand loyalty of established players.  
Global and Localized Pricing Trends

Market data from Sydney and other major tech hubs suggests a shift toward yearly subscriptions as the dominant revenue driver, particularly in Asia-Pacific where yearly plans account for 47% of revenue. For the Australian market, pricing must reflect the localized value of the AFCD (Australian Food Composition Database) integration, which ensures accuracy for local supermarket brands.  
Table 3: Subscription Pricing Architecture (2026 Projections)
Tier Price (Monthly) Price (Annual) Features Included
Freemium $0.00 $0.00 Single photo logs, barcode scanning, 7-day history.
Pro (Standard) $12.99 $64.99 Batch uploads, persistent photos, scale-weight AI.
Family / Team $19.99 $89.99 Up to 6 users, shared custom foods, shared recipes.
Clinical/Coach $29.99 $199.99 Client logging transparency, metabolic analytics.

Higher subscription prices in the Health and Fitness space are paradoxically associated with higher trial conversion rates (9.8% for high-priced vs. 4.3% for low-priced), suggesting that users associate premium pricing with clinical efficacy and better outcomes. A competitive annual price of $64.99 positions the app below MyFitnessPal ($79.99) while offering more advanced AI capabilities.  
SEO and Acquisition Strategy: Capturing 2026 Intent

The transition to "Agentic AI" in search interfaces means that traditional keyword stuffing is no longer effective. Instead, content must be optimized for semantic depth and "Position Zero" snippets. In 2026, user intent in the nutrition space is bifurcated between high-volume informational searches and low-volume, high-intent transactional searches.  
High-Volume Cluster Keywords for 2026

To rank effectively, the app’s web presence must dominate specific 2026 search trends:

    "Agentic Calorie Tracker": Capturing interest in autonomous, reasoning-based assistants.

    "Frictionless Food Logging from Photos": Focusing on the core batch-upload differentiation.

    "Batch Food Photo App": Targeting users frustrated by the one-at-a-time nature of current trackers.

    "AI Weight Estimation from Food Scale": Attracting precision-focused biohackers.

    "Australian Food Database App": Specifically targeting the Sydney/Australian market with NUTTAB/AFCD accuracy.

Table 4: Technical SEO Checklist for Nutrition Applications
Factor Requirement Metric/Tool
E-E-A-T Expert-reviewed by Registered Dietitians (RD).

Medical Reviewer Byline
Semantic Depth 2,000+ word guides on macro-nutrients.

MarketMuse / Clearscope
Freshness Quarterly updates with latest FSANZ/USDA data.

Content Decay Audit
Schema Markup Structured data for recipes and nutrition facts.

Alli AI / Search Console
Image Optimization Descriptive alt-text for ingredient visuals.

AltText.ai / Sitebulb

Technical Architecture: Implementing Multi-Agent Systems with ADK

The core request involves using the Google Agent Development Kit (ADK) to build a sophisticated, modular AI system. ADK is a framework designed to make agent development feel like software development, prioritizing modularity, scalability, and predictable orchestration. Unlike a single complex prompt, ADK allows for the creation of a "Team of Agents," each specialized in a specific sub-task of the nutritional estimation pipeline.  
Agent Hierarchy and Orchestration

The system is structured as a tree, governed by the "Single Parent Rule". A root CoordinatorAgent receives the batch of images and delegates work to sub-agents.

    Vision Specialist Agent: Responsible for image segmentation and ingredient identification. It uses a Vision Transformer (ViT) architecture pre-trained on large-scale datasets like COYO or ImageNet to identify food items in cluttered scenes.

    Scale Analysis Agent: A specialized agent that detects if a scale is present in the image and uses Optical Character Recognition (OCR) to read the displayed weight.

    Volumetric Estimation Agent: Uses depth-sensing logic or shape-from-silhouette techniques to estimate the volume of identified food items in cubic millimeters.

    Database Routing Agent: Maps identified ingredients to specific nutritional databases, prioritizing the Australian Food Composition Database (AFCD) for regional accuracy.

    Hypothesis/Branching Agent: The core logic engine that evaluates different scenarios for weight (e.g., container vs. no container) and selects the most probable outcome.

Table 5: ADK Primitives for Nutrition Logging Workflows
Primitive Usage in App Execution Pattern
ParallelAgent Processing multiple photos from a batch upload simultaneously.

Concurrent execution for speed.
SequentialAgent Moving from item detection to volume estimation to DB lookup.

Linear pipeline with shared state.
LoopAgent Refining ingredient lists based on user feedback or weight mismatch.

Iterative refinement until accuracy met.
session.state Shared whiteboard for agents to pass ingredient data and weights.

Key-value store for cross-agent context.

Implementation of the AI Agent for Precise Calorie Estimation

To achieve "AS accurate as possible" estimation, the agent must reconcile visual data with physical weight constraints. This involves a multi-stage process of segmentation, volumetric modeling, and hypothesis testing.
Stage 1: Batch Image Processing and Segmentation

When a user uploads a batch of images, the ParallelAgent initiates multiple instances of a VisionAgent. Each instance performs semantic segmentation using models like Mask R-CNN or YOLOv8-Seg to isolate individual food regions from the plate and background.

The system must identify the "Container" (Plate, Bowl, or Tray) to establish a reference plane. The depth of the container is a critical parameter for volumetric extraction.  
Stage 2: Volumetric Mass Estimation

The system calculates the volume V of each food item. For 2D images, this is estimated by fitting 3D shape models (e.g., spheres for oranges, cylinders for cups, or prismatic models for irregular items like chicken) to the segmented regions.

The relationship between the 2D image coordinates and the 3D physical space is defined by the camera's intrinsic and extrinsic parameters. The system uses a scale reference—either a known object like a coin or the scale itself—to calculate the pixel-to-millimeter ratio.

The volume of a food item on a base plane is calculated as:
V=∫∫R​[h(x,y)−b(x,y)]dxdy

where R is the segmented food region, h(x,y) is the estimated height of the food at pixel (x,y), and b(x,y) is the height of the base plane (container surface).  
Stage 3: Branching Agent Logic for Scale Weight Validation

The most innovative feature is the "Hypothesis Branching Agent." When a scale weight Wscale​ is detected, the agent must determine if this is a "tared" weight or a "gross" weight.

The agent creates three distinct branches in the session.state:

    Branch A (Tared): Wfood​=Wscale​.

    Branch B (Standard Container): Wfood​=Wscale​−Wplate_standard​.

    Branch C (Custom Subtraction): Wfood​=Wscale​−Vcontainer​×ρcontainer​.

A DecisionAgent then compares the estimated volume Vest​ of the food to the hypothesized weight Wfood​ using food-specific density models (ρfood​):
Error=∣Wfood​−(Vest​×ρfood​)∣

The branch with the lowest error (highest alignment) is selected as the primary entry, but all branches are saved in the artifact history, allowing the user to switch if the AI chooses incorrectly.  
Table 6: Food Density Look-up Examples (NUTTAB/USDA Averages)
Ingredient Density (ρ) g/cm³ Source Variability Factor
Rice (Cooked) 0.81 AFCD Moisture content
Chicken Breast 1.04 USDA Preparation (Grilled vs. Fried)
Avocado 0.92 AFCD Ripeness / Variety
Olive Oil 0.91 USDA Temperature
Broccoli 0.35 AFCD Packing density (Florets vs. Stalk)

User-Centric Design for Retrospective Editability

A significant pain point in current apps is the "locked" nature of AI logs. In the proposed ADK-based system, the conversion from photo to ingredients is treated as a "Draft State" that remains linked to the original visual artifact.  
Ingredient Modification Architecture

The ingredient list is not a static text block but a collection of objects in the session.state. Each object contains:

    ingredient_id: Link to database.

    mass: Estimated weight in grams.

    source_segment: The specific pixels in the photo the ingredient was derived from.

    modification_history: A log of user overrides.

If a user later modifies the mass of an ingredient, the system uses a LoopAgent to check if the new total weight still aligns with the scale image. If a user adds an ingredient (e.g., "I also added 10g of butter"), the system can "re-scan" the image to see if there is visual evidence of a fat layer or sheen that was previously ignored.  
Photo Persistence and Privacy

Photos are stored as high-resolution artifacts using GcsArtifactService (Google Cloud Storage). This ensures that even if a user deletes an entry, the evidence can be restored or re-analyzed with improved models in the future. To maintain privacy, the system implements a "Local Perception" layer where only food-related segments are uploaded to the cloud for processing, while background details are blurred or discarded at the device level.  
Localization for the Australian Market: Sydney and Beyond

For a Sydney-based launch, the app must navigate the nuances of the Australian Food Composition Database (AFCD) versus the standard USDA datasets used by global competitors. Research indicates that using a modified US database for Australians can result in errors of over 10% for macronutrients and significantly higher for micronutrients like calcium and folate.  
NUTTAB/AFCD Integration Strategy

The app will prioritize the AFCD for all Australian users. This is critical for regional specificities:

    Metric Standards: Australia uses the metric system as standard; however, volume measurements for liquids (mL) and solids (g) must be handled with precise density conversions rather than US "cup" approximations.

    Branded Data: The Australian Branded Food Database provides accurate nutrition for local retailers like Coles and Woolworths, which are often missing or incorrectly entered in crowd-sourced apps like MyFitnessPal.

    Produce Variability: Australian Hass avocados and lamb cuts have different fat profiles than those grown in the US. The ADK DatabaseRoutingAgent will use geo-location to select the appropriate data source.

Table 7: Accuracy Discrepancies (USDA vs. AFCD for Australian Foods)
Nutrient USDA Assumption AFCD Reality Potential Impact
Energy (Calories) -5% to +5% Baseline

Group-level stability, individual error.
Total Fat +11% Error Baseline

Significant for Keto/Low-Carb users.
Calcium -15% Error Baseline

High discrepancy in dairy-heavy diets.
Vitamin B12 -13% Error Baseline

Critical for vegan/vegetarian tracking.

Designing the "Winning Choice" for Specific User Personas

To become the "winning choice," the app must move beyond utility toward "Health Intelligence". This is achieved by using ADK's MemoryService to learn user patterns over time.  
Personas and Tailored Workflows

    The High-Performance Athlete (The "Data Junkie"): This user requires the Scale-Aware AI. They use a kitchen scale for every meal and want the app to "prove" its accuracy. The app provides a "Confidence Breakdown" showing how the visual volume matches the scale weight.

    The Busy Parent (The "Batch Logger"): This user takes photos of their meals but doesn't have time to log until the kids are in bed. The app allows them to select 10 photos from their gallery at 9 PM and generates a full day's log in seconds using the ParallelAgent.

    The Clinical Patient (The "Accuracy Dependent"): Managing a condition like diabetes or kidney disease requires precise micronutrient and fiber tracking. The app uses the AFCD/NUTTAB database to ensure laboratory-grade accuracy and allows their dietitian to view the original meal photos for verification.

Conclusion: Actionable Path Toward a Frictionless Nutrition Future

The proposed macro-tracking application represents a paradigm shift in dietary monitoring. By leveraging the Google Agent Development Kit (ADK) to build a multi-agent system, the developer can solve the twin problems of logging friction and estimation inaccuracy.
Strategic Recommendations

    Implement Batch Processing Immediately: Use the ParallelAgent and session.state to enable simultaneous processing of multiple images. This is the primary competitive differentiator against MacroFactor and MyFitnessPal.

    Architect for Persistence: Ensure photos are saved as linked artifacts. This enables the retrospective editability that is missing in current market leaders.

    Deploy the Hypothesis Branching Agent: This is the "killer feature" for biohackers. By assuming both tared and gross weights on a scale and selecting the best fit, the app achieves scientific-grade accuracy.

    Localized Pricing and SEO: Launch with a disruptive annual price of $64.99 and focus SEO efforts on 2026 intent-based keywords like "Agentic Nutrition Tracker" and "Batch Photo Logging".

    Integrate AFCD/NUTTAB: For the Australian market, this is not optional. It provides the accuracy moat required to defeat US-centric competitors.

By synthesizing these technical and market strategies, the application will not merely track calories but will act as a persistent, intelligent, and flexible partner in the user's health journey. The transition from "ask-and-wait" patterns to real-time, agentic collaboration is the future of the category, and ADK provides the robust foundation necessary to build that future today.  
reddit.com
Has anyone tried any of these food tracking apps? : r/loseit - Reddit
Opens in a new window
apps.apple.com
AteMate Food Journal & Diary - App Store - Apple
Opens in a new window
zoe.com
Photologging: ZOE's New Free App Put to the Test
Opens in a new window
macrofactorapp.com
MacroFactor vs. MyFitnessPal: Which Macro Tracking App Wins in ...
Opens in a new window
fitia.app
9 Best Food Tracking Apps of 2025 (Complete Guide)
Opens in a new window
jotform.com
The 5 best AI calorie trackers of 2026 | The Jotform Blog
Opens in a new window
blog.box.com
How AI agents are evolving from following instructions to figuring it out | Box Blog
Opens in a new window
arxiv.org
Agentic Artificial Intelligence (AI): Architectures, Taxonomies, and Evaluation of Large Language Model Agents - arXiv
Opens in a new window
arxiv.org
Chapter 3: Architectures for Building Agentic AI - arXiv
Opens in a new window
pmc.ncbi.nlm.nih.gov
Image-Based Volume Estimation for Food in a Bowl - PMC
Opens in a new window
jmir.org
Advancements in Using AI for Dietary Assessment Based on Food Images: Scoping Review
Opens in a new window
mdpi.com
Validating Accuracy of an Internet-Based Application against USDA Computerized Nutrition Data System for Research on Essential Nutrients among Social-Ethnic Diets for the E-Health Era - MDPI
Opens in a new window
cronometer.com
Introducing Photo Logging: The Easiest Way to Track Your Meals in Cronometer
Opens in a new window
fitia.app
Top 12 Nutrition Tracking Apps (2026) - Fitia
Opens in a new window
digitalepidemiologylab.org
Optimizing Multimodal Prompting for AI-Based Food Volume Estimation
Opens in a new window
pmc.ncbi.nlm.nih.gov
Model-based measurement of food portion size for image-based dietary assessment using 3D/2D registration - PubMed Central
Opens in a new window
reddit.com
How do you guys like MacroFactor compared to MyFitnessPal? - Reddit
Opens in a new window
macrofactorapp.com
Smart Macro Tracker & Diet Coach - MacroFactor app
Opens in a new window
reddit.com
Whats the best tracking app? / Opinions on macrofactor? : r/PetiteFitness - Reddit
Opens in a new window
pmc.ncbi.nlm.nih.gov
Mobile Apps for Dietary and Food Timing Assessment: Evaluation for Use in Clinical Research - NIH
Opens in a new window
oreateai.com
Understanding the Cost of MacroFactor: A Comprehensive Breakdown - Oreate AI Blog
Opens in a new window
reddit.com
Best calorie/ macro tracking app that like offers one time payment for the whole thing and not a subscription? - Reddit
Opens in a new window
nutrisense.io
11 Best Nutrition Tracking Apps of 2026 - Nutrisense Journal
Opens in a new window
google.github.io
Index - Agent Development Kit - Google
Opens in a new window
blog.stackademic.com
Google Agent Development Kit (ADK): Agent Types and Models | by ...
Opens in a new window
arxiv.org
Investigating the Impact of Large-Scale Pre-training on Nutritional Content Estimation from 2D Images - arXiv
Opens in a new window
pmc.ncbi.nlm.nih.gov
Automated Food Weight and Content Estimation Using Computer ...
Opens in a new window
revenuecat.com
State of Subscription Apps 2025 - RevenueCat
Opens in a new window
researchgate.net
Which Australian food composition database should I use? | Request PDF - ResearchGate
Opens in a new window
foodstandards.gov.au
Australian Food Composition Database | Food Standards Australia New Zealand
Opens in a new window
byword.ai
Nutrition Blog SEO Checklist: Complete Optimization Guide (2026)
Opens in a new window
fyresite.com
Top 6 AI SEO Tools to Simplify Your Workflow in 2026 - Fyresite
Opens in a new window
accountabilitynow.net
AI Keyword Research For: Guide to Smarter SEO in 2026 - Accountability Now
Opens in a new window
seranking.com
Top 100 Google Searches in 2025 - SE Ranking
Opens in a new window
explodingtopics.com
Top 100 Google Searches (January 2026) - Exploding Topics
Opens in a new window
reddit.com
Nuttab v Google question : r/ketoaustralia - Reddit
Opens in a new window
apptweak.com
App Store keyword research for ASO: The 2026 step-by-step guide - AppTweak
Opens in a new window
cloud.google.com
Building Collaborative AI: A Developer's Guide to Multi-Agent Systems with ADK
Opens in a new window
sidbharath.com
The Complete Guide to Google's Agent Development Kit (ADK) - Sid Bharath
Opens in a new window
codelabs.developers.google.com
Build Multi-Agent Systems with ADK - Google Codelabs
Opens in a new window
medium.com
Building a multi-agent chatbot with Google ADK | by Tobin Tom ...
Opens in a new window
cloud.google.com
Build multi-agentic systems using Google ADK | Google Cloud Blog
Opens in a new window
google.github.io
Multi-agent systems - Agent Development Kit - Google
Opens in a new window
pmc.ncbi.nlm.nih.gov
Image-Based Food Volume Estimation - PMC - NIH
Opens in a new window
spikeapi.com
Top Nutrition APIs for App Developers in 2026
Opens in a new window
fdc.nal.usda.gov
FoodData Central API Guide
Opens in a new window
tencentcloud.com
How to build the thinking tree of AI Agent? - Tencent Cloud
Opens in a new window
dev.to
Building Dynamic Workflows with Branching Logic in Microsoft Agent Framework Part-IV
Opens in a new window
google.github.io
Parallel agents - Agent Development Kit - Google
Opens in a new window
medium.com
Mastering ADK Workflows: A Developer's Guide to Sequential, Parallel, Loop and Custom Agents | by Hangsik Shin | Medium
Opens in a new window
developers.googleblog.com
Developer's guide to multi-agent patterns in ADK
Opens in a new window
mdpi.com
Automated Food Weight and Content Estimation Using Computer Vision and AI Algorithms: Phase 2 - MDPI
Opens in a new window
pmc.ncbi.nlm.nih.gov
A Food Intake Estimation System Using an Artificial Intelligence–Based Model for Estimating Leftover Hospital Liquid Food in Clinical Environments: Development and Validation Study - NIH
Opens in a new window
scispace.com
Measuring Calorie and Nutrition From Food Image - SciSpace
Opens in a new window
parseur.com
Human-in-the-Loop AI in Document Workflows - Best Practices & Common Pitfalls - Parseur
Opens in a new window
dev.to
Integrating Google ADK to Build Smarter Multimodal AI Agents - DEV Community
Opens in a new window
pmc.ncbi.nlm.nih.gov
Cross-Border Use of Food Databases: Equivalence of US and Australian Databases for Macronutrients - PMC - NIH
Opens in a new window
foodstandards.gov.au
Food and nutrient databases - Food Standards Australia New Zealand
Opens in a new window
codelabs.developers.google.com
ADK with Multimodal Tool Interaction : Part 1 ( Custom Tool with Model Callbacks ) | Google Codelabs
Opens in a new window
adapty.io
App Pricing Index: Global Subscription Pricing Benchmarks - Adapty
Opens in a new window
digiground.com.au
Building Subscription-Based Apps in Australia - DigiGround
Opens in a new window
google.github.io
Custom agents - Agent Development Kit - Google
Opens in a new window
dokumen.pub
Building Applications with AI Agents: Designing and Implementing Multiagent Systems 1
Opens in a new window
whatthefood.io
What the Food: Macro Tracker and Food Calorie Finder
Opens in a new window
datainsightsmarket.com
Exploring Innovation in Calorie Counter Websites and Apps Industry
Opens in a new window
apps.apple.com
Calorie Counter - MyNetDiary - App Store - Apple
Opens in a new window
google.github.io
Part 5. Audio, Images, and Video - Agent Development Kit - Google
Opens in a new window
usg.edu
Earn rewards for a healthy lifestyle
Opens in a new window
anthem.com
Download Anthem's Sydney Health App
Opens in a new window
rc-hr.com
2026 health and wellness programs
Opens in a new window
play.google.com
Sydney Health - Apps on Google Play
Opens in a new window
play.google.com
Sydney Health – Apps on Google Play
Opens in a new window
comet.com
Thread-Level Human-in-the-Loop Feedback for Agent Validation - Comet
Opens in a new window
medium.com
2-Minute ADK: Human-in-the-loop Made Easy | by Bo Yang | Google Cloud - Medium
Opens in a new window
blog.metaphacts.com
Human-in-the-Loop for AI: A Collaborative Future in Research Workflows - metaphacts Blog
Opens in a new window
github.com
An example of an ADK agent performing human-in-the-loop. - GitHub
Opens in a new window
foodstandards.gov.au
Data files | Food Standards Australia New Zealand
Opens in a new window
foodstandards.gov.au
About AFCD - Food Standards Australia New Zealand
Opens in a new window

---

Please follow up with these things:

- For the Australian market, we want to integrate AFCD / NUTTAB databases

- How about for northern America, europe, and other regions where such apps are popular? We need to support all those too

For creating evals for the AI agents to measure quality of food estimations

- Where can we find sizeable datasets that would serve as valuable testing data?

gemini response:

Expanding your application to support international markets requires integrating high-authority, region-specific Food Composition Databases (FCDBs) to ensure accuracy across varied food supplies. Furthermore, building high-quality evaluations for your AI agents necessitates using established, multi-modal datasets that include ground-truth weights and nutritional labels.
Global Food Composition Database Integration

While the Australian market relies on AFCD and NUTTAB, international expansion necessitates a multi-database approach to account for regional differences in nutrient profiles and branded products.
North America

    USDA FoodData Central (USA): This is the gold standard for North American data, offering an API for over 380,000 foods across five categories, including Foundation Foods and a massive Global Branded Foods database. It is frequently updated (quarterly) and provides research-grade micronutrient data.

    Health Canada (CNF): For complete North American coverage, the Canadian Nutrient File (CNF) is typically used alongside the USDA.

Europe and United Kingdom

    CoFID (United Kingdom): McCance and Widdowson’s "Composition of Foods Integrated Dataset" (CoFID) is the official UK reference. It is available as a consolidated electronic dataset managed by Public Health England, covering nearly all foods in the UK supply chain.

    CIQUAL (France): Managed by ANSES, this is one of the most comprehensive tables in Europe, covering over 3,400 foods representative of French consumption, including specific details on individual sugar types.

    EFSA (European Union): The European Food Safety Authority manages the "EU Menu" project, which standardizes food consumption data across 21 countries. A comprehensive Open Access European Food Composition Database is expected to launch in mid-2026.

    EuroFIR: For broader European support, EuroFIR acts as a non-profit association providing access to harmonized food composition data from over 25 European countries.

Asia-Pacific and Global

    Standard Tables of Food Composition (Japan): The STFCJ (2020 edition) includes data for 2,478 food items and is highly specialized for traditional Japanese composite and fermented foods.

    China Food Composition Tables: The 6th edition (2018-2019) provides nutrient data for more than 600 common foods and is available in digital formats for developer integration.

    ASEANFOODS: A regional database covering Southeast Asia (including Thailand, Malaysia, and Singapore), derived from national tables to provide "typical" values for regional diets.

    Open Food Facts (Global Crowdsourced): A collaborative database containing over 2.8 million products from 150+ countries. It is particularly strong for international barcode support and branded product metadata.

AI Evaluation Datasets for Quality Measurement

To evaluate your AI agents’ ability to estimate portion size and caloric density, you should use large-scale datasets that pair RGB/Depth images with laboratory-measured weights.
Dataset Scale Key Features Utility for Your App
Uni-Food 100,000 samples Unified categories, ingredients, and ingredient-level nutrition. Best for scale: Largest balanced dataset for multi-task training.
Nutrition5k 5,000 unique dishes Incremental weighing of ingredients, video streams, and depth images. Best for accuracy: Often cited as the "gold standard" for weight estimation.
Food Portion Benchmark (FPB) 14,083 images 138 food classes with laboratory-measured component weights. Best for weight regression: Features YOLO-style labels with a weight column.
Recipe1M+ 1M recipes / 13M images Massive collection of recipe text and associated imagery. Best for vision: Ideal for training general food recognition models.
MetaFood3D 3D Models Focuses on physically informed 3D food reconstruction from 2D images. Best for volume: Testing the precision of your volumetric agents.
Implementation Note for Evals

Using the Agent Development Kit (ADK) AgentEvaluator, you can run these datasets through your agentic pipeline to measure Mean Absolute Error (MAE) for weight and caloric accuracy. Modern benchmarks, such as the Food AI Leaderboard on Hugging Face, provide standardized test sets for evaluating weight regression heads in multi-task models.
