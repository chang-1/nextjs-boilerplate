import Explorer from "./Explorer";

export const metadata = {
  title: "JEM Predictions & Jacobians",
  description: "α=0.0 vs α=0.1: ImageNet predictions, confidence, and decoder-Jacobian comparisons",
};

export default function Page() {
  return <Explorer />;
}
