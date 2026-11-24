# Global Guidance

## Engineering Taste

- Seek a simpler representation before adding logic. Prefer representations and invariants that make special cases disappear.
- Give each fact one authoritative home and each responsibility a clear owner. Avoid parallel representations that must be kept in sync.
- When writing something intended for human consumption, (comment, commit message, reply to prompt) use as few words as possible. Pick every word meticulously to reduce the volume to a strict minimum. Be down to the point. Less is more.
- Don't feel bound by existing conventions. Consistency with existing work matters, but not with bad practices or implementations; improve those alongside your change instead.

## Documentation

- Keep documentation concise and consistent.
- Only document information worth preserving long-term. Otherwise, tell the user directly rather than documenting it.
- README.md and AGENTS.md are maintained by the user. If you determine that changes are needed, ask the user before modifying these files.

### Architecture Documentation

- Use ARCHITECTURE.md to describe repository-wide architecture, and files under architectures/ to describe the architecture of a single focused topic, while avoiding describing implementation details.
- Treat these documents as summaries of the implementation, not as fixed constraints. Adopt a better representation when appropriate.

### Approve for me

- 在目标和验收标准明确后，持续执行到完成。读取代码、检索仓库、修改实现、运行测试、修复错误和重试，由你自行决定，无需逐步确认，也无需问我是否继续。
- 只有在以下情况才暂停问我：需要改变研究目标、核心技术路线或验收标准；两个方案会显著改变结论或资源投入，且无法根据现有要求取舍；
或者下一步会造成不可逆的数据损失或产生未经授权的费用。提问时给出你的推荐方案和主要取舍。