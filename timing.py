from vllm import LLM, SamplingParams
import os
import time

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

base_prompt = """
**The Timeless Influence of William Shakespeare**

**Introduction**

William Shakespeare, often regarded as the greatest playwright and poet in the English language, has had a profound influence on literature, theater, and the arts. His works have transcended time and space, remaining relevant and impactful across centuries and cultures. This essay delves into the life of Shakespeare, his body of work, the themes he explored, his influence on the English language, and his enduring legacy.

**1. The Life of William Shakespeare**

William Shakespeare was born in Stratford-upon-Avon in April 1564, during the reign of Queen Elizabeth I. The exact date of his birth is unknown, but he was baptized on April 26, 1564, suggesting he was born a few days earlier. His father, John Shakespeare, was a successful glove-maker and local politician, while his mother, Mary Arden, came from a prosperous farming family.

Shakespeare likely attended the Kings New School in Stratford, where he would have received a classical education, studying Latin, literature, and rhetoric. However, there are no records of him attending university, which has led to much speculation about his education and intellectual background.

In 1582, at the age of 18, Shakespeare married Anne Hathaway, who was eight years his senior. They had three children: Susanna, and twins Hamnet and Judith. Hamnet died at the age of 11, a tragedy that may have influenced some of Shakespeare's later works, particularly his exploration of grief and loss in his tragedies.

Shakespeares career in the theater began in London in the late 1580s or early 1590s. He quickly became a prominent figure in the London theater scene, working as an actor, playwright, and shareholder in the Lord Chamberlains Men, a leading theatrical company. His early works were successful, and he soon became the principal playwright of the company.

Shakespeare continued to write and perform until his retirement around 1613, after which he returned to Stratford. He died on April 23, 1616, and was buried in the chancel of Holy Trinity Church in Stratford. His epitaph, reputedly written by himself, warns against moving his bones.

**2. The Body of Work**

Shakespeares body of work includes 39 plays, 154 sonnets, and several narrative poems. His plays are traditionally divided into three categories: tragedies, comedies, and histories. Each category explores different themes and showcases Shakespeare's mastery of language and dramatic structure.

**2.1 Tragedies**

Shakespeare's tragedies are perhaps his most celebrated works. They delve into the human condition, exploring themes of ambition, power, jealousy, betrayal, and mortality. Some of the most famous tragedies include:

- *Hamlet*: This play tells the story of Prince Hamlet's quest for revenge against his uncle, who has murdered Hamlet's father, the king. The play is a profound exploration of madness, indecision, and the complexity of human emotions. The famous soliloquy, "To be or not to be," encapsulates Hamlet's existential crisis.

- *Macbeth*: A tale of ambition and power, *Macbeth* follows the rise and fall of a Scottish nobleman who, spurred by prophecy and his wife's encouragement, murders King Duncan to take the throne. The play is a dark examination of guilt, ambition, and the consequences of moral corruption.

- *Othello*: *Othello* explores themes of jealousy, racism, and betrayal. The play's central character, Othello, is a Moorish general in the Venetian army, whose life unravels due to the manipulations of his envious ensign, Iago. The play's exploration of trust and deception remains relevant today.

- *King Lear*: A powerful tragedy about the consequences of pride and folly, *King Lear* tells the story of an aging king who divides his kingdom among his daughters based on their flattery, leading to his downfall. The play is a poignant exploration of family, loyalty, and the devastating effects of hubris.

**2.2 Comedies**

Shakespeares comedies often involve complex plots, mistaken identities, and witty dialogue. They typically explore themes of love, marriage, and societal norms, often with a light-hearted tone and a happy ending. Some of the most notable comedies include:

- *A Midsummer Night's Dream*: This play is a whimsical exploration of love and magic, set in a mythical Athens and an enchanted forest. The play's multiple interwoven plots involve a love quadrangle, mischievous fairies, and a group of amateur actors. The play is celebrated for its imaginative setting and humorous portrayal of love's complexities.

- *Much Ado About Nothing*: A comedy of wit and misunderstandings, *Much Ado About Nothing* centers on the romantic entanglements of two couples, Beatrice and Benedick, and Claudio and Hero. The play is notable for its sharp dialogue, especially between Beatrice and Benedick, who engage in a "merry war" of words.

- *Twelfth Night*: This play is a festive comedy that explores themes of gender identity and love. The plot revolves around the shipwrecked Viola, who disguises herself as a man and finds herself entangled in a love triangle. *Twelfth Night* is celebrated for its exploration of mistaken identity and the fluidity of gender roles.

- *As You Like It*: Set in the Forest of Arden, *As You Like It* is a pastoral comedy that explores themes of love, exile, and the natural world. The play features one of Shakespeare's most famous speeches, "All the world's a stage," and is known for its celebration of love in its various forms.

**2.3 Histories**

Shakespeares history plays are dramatizations of historical events, primarily focusing on the lives of English kings. These plays explore themes of power, leadership, and the complexities of political life. Some of the most significant history plays include:

- *Richard III*: This play portrays the Machiavellian rise to power of Richard III, who manipulates and murders his way to the throne. The play is a study of ambition, tyranny, and the corrupting influence of power. Richard III is one of Shakespeare's most compelling villains, known for his cunning and ruthless pursuit of the crown.

- *Henry IV, Part 1 and Part 2*: These plays follow the reign of King Henry IV and the maturation of his son, Prince Hal, who will eventually become King Henry V. The plays explore themes of honor, rebellion, and the responsibilities of kingship. They also introduce the character of Falstaff, one of Shakespeare's most beloved comic figures.

- *Henry V*: A continuation of the Henry IV plays, *Henry V* focuses on the young king's military campaign in France, culminating in the Battle of Agincourt. The play is a patriotic celebration of English valor and leadership, with the famous "St. Crispin's Day" speech rallying the troops.

- *Richard II*: This play examines the downfall of King Richard II, who is deposed by Henry Bolingbroke, later King Henry IV. The play is a meditation on kingship, identity, and the divine right of kings. Richard II's introspective soliloquies offer a poignant exploration of the fragility of power.

**3. Themes in Shakespeare's Work**

Shakespeare's works are renowned for their exploration of universal themes that continue to resonate with audiences today. His ability to delve into the complexities of human nature has ensured his plays' lasting relevance. Some of the key themes in his work include:

**3.1 The Human Condition**

Shakespeare's plays are deeply concerned with the human condition, exploring the full range of human emotions and experiences. His characters grapple with love, hate, jealousy, ambition, power, and mortality. Through their struggles, Shakespeare reveals the complexities and contradictions of human nature. His tragedies, in particular, offer profound insights into the darker aspects of the human psyche.

**3.2 Power and Ambition**

The theme of power and ambition is central to many of Shakespeare's plays, especially his tragedies and histories. Characters like Macbeth, Richard III, and Julius Caesar are consumed by their desire for power, leading to their eventual downfall. Shakespeare explores the corrupting influence of power and the moral dilemmas faced by those who seek it.

**3.3 Love and Relationships**

Love is a recurring theme in Shakespeare's comedies, tragedies, and sonnets. He explores various forms of love, from romantic and familial love to unrequited and obsessive love. Shakespeare's portrayal of love is often complex and multifaceted, highlighting both its joys and sorrows. His comedies typically end with marriages and celebrations, while his tragedies often depict love's destructive potential.

**3.4 Identity and Disguise**

The theme of identity and disguise is prominent in many of Shakespeare's plays, particularly his comedies. Characters often assume false identities or disguise themselves, leading to misunderstandings and comedic situations. Plays like *Twelfth Night*, *As You Like It*, and *The Merchant of Venice* explore the fluidity of identity and the ways in which people present themselves to the world.

**3.5 Fate and Free Will**

Shakespeare often explores the tension between fate and free will in his plays. Characters like Macbeth and Oedipus grapple with the idea of destiny and whether their actions are predetermined or a result of their choices. This theme is particularly evident in *Hamlet*, where the protagonist struggles with the idea of fate and his role in avenging his father's death.

**4. Shakespeare's Influence on the English Language**

Shakespeare's influence on the English language is unparalleled. He is credited with coining or popularizing thousands of words and phrases that are still in use today. His inventive use of language, wordplay, and metaphor has enriched the English lexicon and shaped the way we speak and write.

Answer the following question in 200 words:

"""

questions = [
    "How do Shakespeare's tragedies explore the complexities of human nature, and what do they reveal about the darker aspects of the human experience?",
    "In what ways do Shakespeare's plays reflect the social, political, and cultural context of Elizabethan England, and how do they remain relevant to modern audiences?",
    "What role does the theme of identity play in Shakespeare's comedies, and how does it contribute to the humor and complexity of his characters?",
    "How does Shakespeare use the supernatural in his plays, such as Macbeth and The Tempest, to explore themes of fate, power, and the unknown?",
    "What is the significance of gender roles and cross-dressing in Shakespeare's comedies, such as Twelfth Night and As You Like It, and how do these elements challenge or reinforce societal norms?",
    "How does Shakespeare use the motif of madness in plays like Hamlet and King Lear to reflect the characters' inner turmoil and societal pressures?",
    "What is the role of fate versus free will in Shakespeare's tragedies, and how do characters like Macbeth and Othello grapple with these concepts?",
    "How do Shakespeares history plays, such as Henry V and Richard III, portray the responsibilities and burdens of leadership and kingship?",
    "In what ways does Shakespeares use of language, including wordplay and puns, enhance the themes and character dynamics in his plays?",
    "How does Shakespeare explore the theme of revenge in plays like Hamlet and Titus Andronicus, and what does it reveal about justice and morality?",
    "What role do women play in Shakespeare's tragedies, and how do characters like Lady Macbeth and Desdemona challenge or conform to the expectations of their time?",
    "How does Shakespeare use the setting of the natural world in plays like A Midsummer Nights Dream and As You Like It to explore themes of love, identity, and transformation?",
    "In what ways do Shakespeare's sonnets explore the passage of time and the nature of beauty, and how do these themes compare to those in his plays?",
    "How does Shakespeare depict the theme of betrayal in plays like Julius Caesar and Othello, and what impact does it have on the characters and plot?",
    "What is the significance of dreams and visions in Shakespeare's plays, such as in Richard III and The Tempest, and how do they contribute to the narrative?",
    "How do Shakespeare's plays address the concept of honor, and how do characters like Falstaff in Henry IV challenge traditional notions of honor?",
    "In what ways does Shakespeare's portrayal of love in Romeo and Juliet differ from his depiction of love in his comedies, such as Much Ado About Nothing?",
    "How does Shakespeare explore the theme of power and its corrupting influence in plays like Macbeth and Julius Caesar?",
    "What role do minor characters play in Shakespeares plays, and how do they contribute to the overall themes and plot development?",
    "How does Shakespeare use irony in his plays, and what effect does it have on the audiences understanding of the characters and plot?"
]


prompts = [base_prompt+q for q in questions]

llm = LLM(model="meta-llama/Meta-Llama-3-8B", enable_prefix_caching=True, use_v2_block_manager=True, enforce_eager=True)

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=1000, temperature=0.0))
end_time = time.perf_counter()

for o in outputs:
    print(o.outputs)

print("PERFORMANCE:", end_time-start_time)


