Source: https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-sonic-next-generation-speech-to-speech-model-for-conversational-ai/

# Introducing Amazon Nova 2 Sonic: Our new speech-to-speech model for conversational AI

|  |
| --- |
| [Voiced by Polly](https://aws.amazon.com/polly/) |

Today, we are announcing the general availability of [Amazon Nova 2 Sonic](https://aws.amazon.com/ai/generative-ai/nova/speech/), a speech-to-speech foundation model that brings natural, real-time voice conversations to your applications. The model delivers industry-leading conversational quality, pricing, and best-in-class speech understanding for developers to build voice applications.

Amazon has been a leader in voice-based technology for over a decade, and earlier this year, we [introduced the first generation of Nova Sonic](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-sonic-human-like-voice-conversations-for-generative-ai-applications/) to solve the fundamental challenge of creating truly fluid voice interactions—preserving the acoustic context to adapt the speech response to not just to what the user said but how they said it. With Nova 2 Sonic, we have built on that foundation by making the model more capable and more accessible improving model intelligence and agentic capabilities, expanding language support, and adding a broad range of new features to provide more intuitive, human-like voice interactions.

Nova 2 Sonic delivers expressive voices, masculine and feminine voices in each of the supported languages with native expressivity, natural turn-taking, and seamless handling of user interruptions. Human preference evaluations show that listeners consistently favor Nova 2 Sonic output over other leading models for overall listening experience.

[![Amazon Nova 2 Sonic conversation quality](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2025/12/02/nova-2-sonic-conversation-quality-1.png)](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2025/12/02/nova-2-sonic-conversation-quality-1.png)

Nova 2 Sonic delivers strong intelligence and more reliable agentic behavior, supported by improvements across key evaluation benchmarks. The model outperforms other leading conversational AI models on [Big Bench Audio](https://huggingface.co/blog/big-bench-audio-release), an evaluation dataset for assessing reasoning capabilities with audio input. Its [BFCL benchmark](https://www.salesforce.com/blog/bfcl-audio-benchmark/) score highlights more accurate and consistent function calling, while [ComplexFuncBench](https://arxiv.org/abs/2501.10132) results reflect better handling of multi-step, constraint-heavy tasks. We used [Common Voice](https://commonvoice.mozilla.org/) to demonstrate improved automatic speech recognition (ASR) accuracy, and [Instruction-Following Evaluation (IFEval)](https://arxiv.org/abs/2311.07911) to show higher accuracy in following detailed, structured instructions.

[![](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2025/12/03/Nova-Sonic-Speech-Understanding-Core-Intelligence-BOLDED-COMPRESSED_corrected120325.png)](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2025/12/03/Nova-Sonic-Speech-Understanding-Core-Intelligence-BOLDED-COMPRESSED_corrected120325.png)

**Improved speech understanding**
The underlying speech recognition capabilities have been significantly enhanced in Nova 2 Sonic. The model now handles alphanumeric inputs, short utterances, and 8KHz telephony speech input with improved accuracy. It’s also more robust when dealing with different accents and background noise—critical for real-world deployment scenarios.

**Expanded global reach with polyglot voices**
One of the most significant updates in Nova 2 Sonic is expanded language support. Beyond the original English, French, Italian, German, and Spanish languages, Nova 2 Sonic now supports Portuguese and Hindi.

Beyond supporting multiple languages, Nova 2 Sonic introduces polyglot voices—individual voices that can switch between languages within the same conversation. The Tiffany voice, for example, can now speak all supported languages fluidly in a single interaction. This offers advanced [code-switching](https://en.wikipedia.org/wiki/Code-switching) (the linguistic term for mixing languages within sentences) capabilities that handle mixed-language sentences naturally. For example, to respond back in user’s preferred language when the user switches languages from one turn to the next in the same conversational dialog.

For developers, this means you can build applications that serve global audiences without needing separate voice models for each language. A customer support application could handle a dialogue that starts in English and switches to Spanish mid-conversation, maintaining the same flow and voice characteristics throughout.

**Natural turn-taking**
Turn-taking has been enhanced with configurable voice activity detection sensitivity. Developers can set this to high, medium, or low depending on their use case. High sensitivity optimizes for the fastest response times, while low sensitivity gives users more time to complete their thoughts. This is useful, for example, for educational applications or to provide conversational AI for users with different communication preferences.

**Seamless crossmodal interactions**
With crossmodal support, users can switch between text and voice input within the same session. This is valuable for applications where users might want to speak some requests and type others—perhaps speaking a quick question but typing a complex address or technical specification.

The implementation maintains context across modalities, so a user could start a conversation by typing a question, receive a spoken response, then continue with voice input without losing the current thread. This creates more fluid, flexible interactions that adapt to how users actually want to communicate.

You can now use the crossmodal feature to prompt the model in text to enunciate a personalized welcome greeting at the beginning of the dialog (to speak first), or use text metadata representing keypad tones to navigate interactive voice response (IVR) applications. For example, when making an outbound call with Nova 2 Sonic to make a reservation on behalf of the user or leave a voicemail.

**Advanced multiagent capabilities**
Nova 2 Sonic introduces asynchronous tool calling that improves how speech-based conversational AI handles complex, multi-step tasks. When the model needs to call external tools or services, it doesn’t pause but continues to respond to new user input while tools run in the background.

Here’s how this works in practice: A user might ask “What’s the weather like?” and immediately follow up with “What is next on my task list?” Nova 2 Sonic processes all these requests, responds to the question immediately, and then provides the weather and task information as the respective tools return their results.

Just as we naturally handle multiple concurrent topics in a discussion, this capability supports sophisticated interactions that can manage multiple unrelated tasks while maintaining engagement and responsiveness.

**Enhanced telephony and platform integration**
Recognizing that many conversational AI applications need to work across different communication channels, Nova 2 Sonic now includes direct integration with leading telephony providers including [Amazon Connect](https://aws.amazon.com/connect/), [Vonage](https://www.vonage.com/), [Twilio](https://www.twilio.com/), and [Audiocodes](https://www.audiocodes.com/), and media platforms like [LiveKit](https://livekit.io/) and [Pipecat](https://www.pipecat.ai/).

These integrations handle the complex technical requirements of phone-based interactions, such as audio codec optimization, session lifecycle management, bidirectional input/output event handling, and the acoustic challenges of telephony systems. For developers, this means you can deploy Nova 2 Sonic-powered applications directly into existing call center infrastructure or build new phone-based services without managing the underlying telephony complexity.

**Getting started with Nova 2 Sonic**
Nova 2 Sonic is available through [Amazon Bedrock](https://aws.amazon.com/bedrock) using the model ID `amazon.nova-2-sonic-v1:0`. If you’re already using Nova Sonic in your applications, updating to the new version is straightforward—simply update the model ID in your existing code, and your application will immediately benefit from the enhancements that don’t require additional configurations.

The model uses the same bidirectional streaming API as the original Nova Sonic, so your existing integration patterns and event handling code will continue to work. New features like crossmodal input and configurable turn taking are available through additional parameters and events that you can adopt incrementally.

To get started with the code examples for multiple programming languages, see the [Amazon Nova Sonic Speech-to-Speech Model Samples](https://github.com/aws-samples/amazon-nova-samples/tree/main/speech-to-speech).

**Things to know**
[Amazon Nova 2 Sonic](https://aws.amazon.com/ai/generative-ai/nova/speech/) is available in the US East (N. Virginia), US West (Oregon), and Asia Pacific (Tokyo) [AWS Region](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/). For Regional availability and future roadmap, visit [AWS Capabilities by Region](https://builder.aws.com/capabilities/).

Nova 2 Sonic maintains the industry-leading price performance and low latency of the original Nova Sonic. Pricing information is available on the [Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/).

The model supports the same robust security and compliance features as other Amazon Bedrock models, including encryption in transit and at rest, [VPC endpoints](https://docs.aws.amazon.com/vpc/latest/privatelink/create-interface-endpoint.html), and integration with [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) for fine-grained access control.

Nova 2 Sonic includes built-in safety controls to promote [responsible AI](https://aws.amazon.com/ai/responsible-ai/?trk=e61dee65-4ce8-4738-84db-75305c9cd4fe&sc_channel=el) use, with content moderation capabilities that help maintain appropriate outputs across a wide range of applications.

To learn more about Amazon Nova 2 Sonic and start building, check out the [Nova Sonic section of the Amazon Nova User Guide](https://docs.aws.amazon.com/nova/latest/userguide/speech.html) for detailed implementation guidance.

**Learn more**
[Amazon Nova main product page](https://aws.amazon.com/nova/)

**More Amazon Nova launches:**
[Introducing Amazon Nova 2 Lite, a fast, cost-effective reasoning model](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-lite-a-fast-cost-effective-reasoning-model)
[Introducing Amazon Nova Forge: Build your own frontier models using Nova](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-forge-build-your-own-frontier-models-using-nova)
[Introducing Amazon Nova 2 Omni in Preview](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-nova-2-omni-preview/)
[Build reliable AI agents for UI workflow automation with Amazon Nova Act, now](https://aws.amazon.com/blogs/aws/build-reliable-ai-agents-for-ui-workflow-automation-with-amazon-nova-act-now-generally-available)

12/12/25 – Editor’s Note: Rollback of Europe (Stockholm) region

— [Danilo](https://x.com/danilop)
