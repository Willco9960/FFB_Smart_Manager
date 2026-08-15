# Security and responsible use

## Scope

This project is a local research and decision-support application. It is not currently a production service and should not be trusted to submit unattended transactions to a fantasy platform.

## Secrets

Never commit or paste into source files:

- ESPN cookies, `SWID`, or `espn_s2` values.
- API keys or authentication tokens.
- Private league identifiers or exported account data.
- Local model credentials or server secrets.

Use an ignored `.env` file for local configuration. `.env.example` is intentionally non-secret.

## Platform access

Any future ESPN or other platform connector should begin read-only, respect the platform’s terms, and require explicit human approval before changing a roster, submitting a waiver, or proposing a trade. Credentials should be stored locally and encrypted where possible.

## Model safety

- The league engine must validate every proposed action.
- The language-model coach may explain structured recommendations but must not override scoring, ownership, or legality checks.
- Training reports must identify data leakage risks, simplified rules, and failed validation gates.
- A failed value-model validation should fall back to the transparent genome/rules path.

## Reporting a vulnerability

Do not open a public issue containing credentials or private league data. Remove the secret, preserve a minimal reproduction if possible, and contact the repository owner privately through the project’s hosting platform.
