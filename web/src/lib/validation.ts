import { z } from "zod";

export const signUpSchema = z.object({
  email: z.string().email(),
  username: z
    .string()
    .min(3)
    .max(24)
    .regex(/^[a-zA-Z0-9_]+$/, "Username can only contain letters, numbers, and underscore."),
  password: z.string().min(8).max(128),
});
export type SignUpInput = z.infer<typeof signUpSchema>;

export const signInSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
});
export type SignInInput = z.infer<typeof signInSchema>;

export const createScanSchema = z.object({
  title: z.string().min(2).max(120),
  sport: z.string().min(2).max(40),
  year: z.string().optional().nullable(),
  brand: z.string().max(80).optional().nullable(),
  setName: z.string().max(80).optional().nullable(),
  playerName: z.string().min(2).max(80),
  cardNumber: z.string().max(40).optional().nullable(),
  serialNumber: z.string().max(40).optional().nullable(),
  disclaimerAccepted: z.boolean().refine((v) => v, {
    message: "Disclaimer must be accepted.",
  }),
});
export type CreateScanInput = z.infer<typeof createScanSchema>;

export const createScanFormSchema = z.object({
  title: z.string().min(2).max(120),
  sport: z.string().min(2).max(40),
  year: z.string().optional(),
  brand: z.string().optional(),
  setName: z.string().optional(),
  playerName: z.string().min(2).max(80),
  cardNumber: z.string().optional(),
  serialNumber: z.string().optional(),
  disclaimerAccepted: z.boolean().refine((value) => value, {
    message:
      "You must acknowledge that RAWIFY is AI-assisted verification and not an official grade.",
  }),
});
export type CreateScanFormInput = z.infer<typeof createScanFormSchema>;

export const uploadImageSchema = z.object({
  type: z.enum([
    "front_straight",
    "back_straight",
    "front_angle_left",
    "front_angle_right",
    "back_angle_left",
    "back_angle_right",
    "corner_macro_top_left",
    "corner_macro_top_right",
    "corner_macro_bottom_left",
    "corner_macro_bottom_right",
    "other",
  ]),
  originalUrl: z.string().url(),
  width: z.number().int().positive().optional(),
  height: z.number().int().positive().optional(),
  blurScore: z.number().min(0).max(100).optional(),
  glareScore: z.number().min(0).max(100).optional(),
  alignmentScore: z.number().min(0).max(100).optional(),
  accepted: z.boolean().default(false),
});

export const createListingSchema = z.object({
  cardScanId: z.string().min(8),
  askingPrice: z.number().positive(),
  description: z.string().min(8).max(1000),
  status: z.enum(["draft", "active", "sold", "archived"]).default("draft"),
});
