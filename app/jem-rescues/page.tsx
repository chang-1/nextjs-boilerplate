import Explorer from "./Explorer";

export const metadata = {
  title: "JEM Predictions & Attributions",
  description: "α=0.0 vs α=0.1: ImageNet predictions, confidence, and input-gradient and decoder-projection comparisons",
};

export default function Page() {
  return <Explorer />;
}
