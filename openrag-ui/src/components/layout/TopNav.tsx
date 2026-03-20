
interface TopNavProps {
  title: string;
}

export function TopNav({ title }: TopNavProps) {
  return (
    <div className="top-bar">
      <div className="top-bar-title">{title}</div>
      <div className="top-bar-actions flex items-center gap-4">
        <button className="btn btn-ghost">Documentation</button>
        <button className="btn btn-primary">Deploy</button>
      </div>
    </div>
  );
}
