export type ScanStatus = "draft" | "processing" | "completed" | "failed" | "listed";
export type ListingStatus = "draft" | "active" | "sold" | "archived";

export type ImageType =
  | "front_straight"
  | "back_straight"
  | "front_angle_left"
  | "front_angle_right"
  | "back_angle_left"
  | "back_angle_right"
  | "corner_macro_top_left"
  | "corner_macro_top_right"
  | "corner_macro_bottom_left"
  | "corner_macro_bottom_right"
  | "other";

export type OverallCategory =
  | "gem_candidate"
  | "strong_raw"
  | "borderline"
  | "visible_risk";

export interface UserSession {
  id: string;
  email: string;
  username: string;
  profileImage?: string | null;
}

export interface ImageQualityResult {
  blurScore: number;
  glareScore: number;
  alignmentScore: number;
  lightingScore: number;
  edgesVisible: boolean;
  accepted: boolean;
  prompts: string[];
}

export interface AnalysisResult {
  success: boolean;
  imageConfidence: number;
  overallScore: number;
  overallCategory: "Gem Candidate" | "Strong Raw" | "Borderline" | "Visible Risk";
  flags: string[];
  subscores: {
    centering: number;
    corners: number;
    edges: number;
    surface: number;
  };
  details: {
    centering: Record<string, string | number>;
    corners: Record<string, string | number>;
    edges: Record<string, string | number>;
    surface: Record<string, string | number>;
  };
  disclaimer: string;
}

export interface PublicListingSummary {
  id: string;
  askingPrice: number;
  description: string;
  status: ListingStatus;
  createdAt: string;
  scan: {
    id: string;
    title: string;
    sport: string;
    playerName: string;
    overallCategory?: string | null;
    overallScore?: number | null;
    imageConfidence?: number | null;
    shareToken?: string;
  };
  seller: {
    username: string;
    profileImage?: string | null;
  };
}

export interface ScanDetailImage {
  id: string;
  type: ImageType;
  accepted: boolean;
  originalUrl: string;
}

export interface ScanDetailDto {
  id: string;
  title: string;
  shareToken: string;
  status: ScanStatus;
  overallCategory?: string | null;
  overallScore?: number | null;
  imageConfidence?: number | null;
  images: ScanDetailImage[];
}

export interface PublicReportPayload {
  title: string;
  sport: string;
  playerName: string;
  overallCategory?: string | null;
  overallScore?: number | null;
  imageConfidence?: number | null;
  scanDate: string;
  disclaimer: string;
  trustCopy: string[];
  flags?: string[];
  subscores?: {
    centering: number;
    corners: number;
    edges: number;
    surface: number;
  };
  images: {
    id: string;
    type: string;
    originalUrl: string;
    accepted: boolean;
  }[];
}
