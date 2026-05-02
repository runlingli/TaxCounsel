"""
Evaluation dataset: 10 questions drawn from IRS Pub 17 (2025 edition).

All answers verified against the actual PDF text.
"""

EVAL_SAMPLES = [
    {
        "question": "What is the standard deduction for a single filer in 2025?",
        "ground_truth": (
            "For 2025, the standard deduction for a single filer is $15,000. "
            "An additional $2,000 applies if you are 65 or older or blind, "
            "so the total reaches $17,000 for one qualifying condition, $19,000 for two. "
            "These amounts appear in the Standard Deduction Worksheet in IRS Pub 17."
        ),
    },
    {
        "question": "What is the filing deadline for individual federal income tax returns?",
        "ground_truth": (
            "The due date for filing your federal income tax return is April 15 of the year "
            "following the tax year. If April 15 falls on a Saturday, Sunday, or legal holiday, "
            "the due date is the next business day. You can request an automatic 6-month "
            "extension by filing Form 4868 by the original due date."
        ),
    },
    {
        "question": "What is the penalty for filing a tax return late?",
        "ground_truth": (
            "The failure-to-file penalty is usually 5% of unpaid tax for each month or "
            "part of a month the return is late, up to 25% of unpaid taxes. "
            "If the return is more than 60 days late, the minimum penalty is the smaller of "
            "$510 or the unpaid tax amount. The penalty is reduced if you also owe the "
            "failure-to-pay penalty for the same period."
        ),
    },
    {
        "question": "Is Social Security income taxable?",
        "ground_truth": (
            "Up to 85% of your Social Security benefits may be taxable depending on your "
            "combined income — adjusted gross income plus nontaxable interest plus half of "
            "your Social Security benefits. If combined income is below $25,000 for a single "
            "filer, benefits are not taxable. Between $25,000 and $34,000, up to 50% may be "
            "taxable. Above $34,000, up to 85% may be taxable."
        ),
    },
    {
        "question": "What is the earned income tax credit?",
        "ground_truth": (
            "The Earned Income Tax Credit (EITC or EIC) is a refundable tax credit for "
            "workers with low to moderate income. The amount depends on your earned income, "
            "filing status, and number of qualifying children. You must have earned income "
            "and meet income limits to qualify. Workers without children may also qualify "
            "if they meet age and residency requirements."
        ),
    },
    {
        "question": "What records should I keep for tax purposes and for how long?",
        "ground_truth": (
            "Keep tax records as long as the IRS can assess additional tax — generally 3 years "
            "from the filing date. Keep records 6 years if you underreported income by more "
            "than 25%. Keep records indefinitely if you filed a fraudulent return. "
            "Employment tax records must be kept at least 4 years. Property records should "
            "be kept until the property is sold plus the statute of limitations period."
        ),
    },
    {
        "question": "How do I report tip income?",
        "ground_truth": (
            "You must report all tip income you receive to your employer if tips total $20 or "
            "more in a calendar month. Your employer includes reported tips on your W-2. "
            "You must also report all tips on your tax return. "
            "Use Form 4137 to calculate Social Security and Medicare tax on unreported tips. "
            "Keep a daily tip record using Form 4070A or a similar diary."
        ),
    },
    {
        "question": "What is the child tax credit?",
        "ground_truth": (
            "The Child Tax Credit is up to $2,000 per qualifying child under age 17 at the "
            "end of the tax year. Up to $1,700 of the credit may be refundable as the "
            "Additional Child Tax Credit. The credit phases out at $400,000 of modified AGI "
            "for married filing jointly, and $200,000 for all other filers."
        ),
    },
    {
        "question": "Do I have to pay taxes on unemployment compensation?",
        "ground_truth": (
            "Yes, unemployment compensation is fully taxable as ordinary income. "
            "You should receive Form 1099-G showing the total amount paid. "
            "You may choose to have 10% federal income tax withheld by filing Form W-4V "
            "with your state unemployment agency."
        ),
    },
    {
        "question": "What is the standard mileage rate for business driving?",
        "ground_truth": (
            "The IRS publishes a standard mileage rate each year for business use of a "
            "personal vehicle. You can use the standard mileage rate instead of tracking "
            "actual car expenses. You must choose between the standard mileage rate and "
            "actual expenses in the first year you use the car for business. "
            "The rate is announced in an IRS Notice each year."
        ),
    },
]
